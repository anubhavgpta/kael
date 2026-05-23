// -----------------------------------------------------------------------
// Module : attention_ctrl
// Project: Kael -- Attention Accelerator (Circle Inference Silicon IP)
// Purpose: Top-level controller, direct streaming, no FIFO/replay
// -----------------------------------------------------------------------

module attention_ctrl #(
    parameter HEAD_DIM=64, DATA_WIDTH=16, NUM_SESSIONS=8, MAX_BATCH=8
)(
    input  logic clk, rst_n,
    input  logic [15:0] q_data,
    input  logic [5:0]  q_addr,
    input  logic [2:0]  q_batch_id,
    input  logic        q_valid,
    input  logic [2:0]  batch_size,
    input  logic [2:0]  session_id,
    input  logic [15:0] token_start, token_end,
    input  logic        attn_start,
    output logic        rd_req,
    output logic [2:0]  rd_session_id,
    output logic [15:0] rd_token_start, rd_token_end,
    input  logic [15:0] rd_k_data, rd_v_data,
    input  logic        rd_valid, rd_last, rd_busy,
    output logic [15:0] ctx_out,
    output logic [2:0]  ctx_batch_id,
    output logic        ctx_valid, ctx_last,
    output logic        attn_done, attn_busy
);

    localparam BATCH_BITS = 3;
    localparam IDX_BITS   = 6;

    typedef enum logic [2:0] {IDLE, FETCH_KV, K_START, STREAM, OUTPUT} state_t;

    state_t state;

    logic [2:0] sess_reg;
    logic [15:0] ts_reg, te_reg;
    logic [3:0] active_batch;
    logic [IDX_BITS-1:0] elem_idx;
    logic need_kstart;
    logic stream_done;
    logic rd_busy_seen;

    logic seq_start_pulse;
    logic k_start_pulse;
    logic k_valid_pulse;
    logic v_valid_pulse;
    logic v_seq_last;
    logic score_seq_last_pulse;
    logic dot_last_pending;
    logic dot_last_d1;
    logic score_last_d1;

    logic [31:0] dot_result_b [0:MAX_BATCH-1];
    logic        dot_valid_b  [0:MAX_BATCH-1];
    logic [15:0] score_b      [0:MAX_BATCH-1];
    logic        score_valid_b[0:MAX_BATCH-1];
    logic [15:0] weight_b     [0:MAX_BATCH-1];
    logic        weight_valid_b [0:MAX_BATCH-1];
    logic        rescale_valid_b [0:MAX_BATCH-1];
    logic [15:0] rescale_factor_b[0:MAX_BATCH-1];
    logic [31:0] denom_b      [0:MAX_BATCH-1];
    logic        denom_valid_b[0:MAX_BATCH-1];
    logic [15:0] ctx_out_b    [0:MAX_BATCH-1];
    logic        ctx_valid_b  [0:MAX_BATCH-1];
    logic        ctx_last_b   [0:MAX_BATCH-1];
    logic        vacc_stall_b [0:MAX_BATCH-1];

    logic        ctx_done     [0:MAX_BATCH-1];
    logic [15:0] ctx_buf      [0:MAX_BATCH-1][0:HEAD_DIM-1];
    logic [IDX_BITS-1:0] ctx_cap_idx [0:MAX_BATCH-1];
    logic [15:0] v_buf        [0:1][0:HEAD_DIM-1];
    logic [31:0] ctx_calc     [0:MAX_BATCH-1][0:HEAD_DIM-1];
    logic        v_wr_buf;
    logic        weight_rd_buf[0:MAX_BATCH-1];
    logic [15:0] weight_count [0:MAX_BATCH-1];
    logic [31:0] weight_sum   [0:MAX_BATCH-1];
    logic [15:0] token_count_reg;
    logic [BATCH_BITS-1:0] out_batch;
    logic [IDX_BITS-1:0] out_idx;

    logic any_stall;
    logic all_done;
    logic active_b [0:MAX_BATCH-1];
    logic [63:0] norm_calc;

    genvar b;
    generate
        for (b = 0; b < MAX_BATCH; b = b + 1) begin : gen_pipe
            assign active_b[b] = (b < active_batch);

            qk_dot_engine #(
                .HEAD_DIM(64),
                .DATA_WIDTH(16),
                .PE_COUNT(8),
                .ACC_WIDTH(32),
                .MAX_BATCH(8)
            ) u_dot (
                .clk(clk),
                .rst_n(rst_n),
                .q_data(q_data),
                .q_valid(q_valid),
                .q_addr(q_addr),
                .q_batch_id(q_batch_id),
                .k_data(rd_k_data),
                .k_valid(k_valid_pulse && active_b[b]),
                .k_start(k_start_pulse && active_b[b]),
                .k_batch_id(b[2:0]),
                .dot_result(dot_result_b[b]),
                .dot_valid(dot_valid_b[b])
            );

            score_scaler #(
                .IN_WIDTH(32),
                .OUT_WIDTH(16)
            ) u_scale (
                .clk(clk),
                .rst_n(rst_n),
                .dot_in(dot_result_b[b]),
                .dot_valid(dot_valid_b[b] && active_b[b]),
                .score_out(score_b[b]),
                .score_valid(score_valid_b[b])
            );

            softmax_engine #(
                .SCORE_WIDTH(16),
                .ACC_WIDTH(32),
                .EXP_WIDTH(16)
            ) u_softmax (
                .clk(clk),
                .rst_n(rst_n),
                .score_in(score_b[b]),
                .score_valid(score_valid_b[b] && active_b[b]),
                .seq_last(score_seq_last_pulse && active_b[b]),
                .seq_start(seq_start_pulse && active_b[b]),
                .weight_out(weight_b[b]),
                .weight_valid(weight_valid_b[b]),
                .rescale_valid(rescale_valid_b[b]),
                .rescale_factor(rescale_factor_b[b]),
                .denom_out(denom_b[b]),
                .denom_valid(denom_valid_b[b])
            );

            v_accumulator #(
                .HEAD_DIM(64),
                .DATA_WIDTH(16),
                .WEIGHT_WIDTH(16),
                .ACC_WIDTH(32)
            ) u_vacc (
                .clk(clk),
                .rst_n(rst_n),
                .v_data(rd_v_data),
                .v_valid(v_valid_pulse && active_b[b]),
                .weight_in(weight_b[b]),
                .weight_valid(weight_valid_b[b] && active_b[b]),
                .rescale_valid(rescale_valid_b[b] && active_b[b]),
                .rescale_factor(rescale_factor_b[b]),
                .seq_start(seq_start_pulse && active_b[b]),
                .seq_last(v_seq_last && active_b[b]),
                .denom_in(denom_b[b]),
                .denom_valid(denom_valid_b[b] && active_b[b]),
                .ctx_out(ctx_out_b[b]),
                .ctx_valid(ctx_valid_b[b]),
                .ctx_last(ctx_last_b[b]),
                .stall(vacc_stall_b[b])
            );
        end
    endgenerate

    integer comb_i;
    integer seq_i;
    integer seq_j;

    always_comb begin
        any_stall = 1'b0;
        all_done = 1'b1;
        for (comb_i = 0; comb_i < MAX_BATCH; comb_i = comb_i + 1) begin
            if (comb_i < active_batch) begin
                any_stall = any_stall | vacc_stall_b[comb_i];
                all_done = all_done & ctx_done[comb_i];
            end
        end
        if (weight_sum[out_batch] != 32'd0) begin
            norm_calc = ({32'd0, ctx_calc[out_batch][out_idx]} << 15) / weight_sum[out_batch];
        end else begin
            norm_calc = 64'd0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            sess_reg <= 3'd0;
            ts_reg <= 16'd0;
            te_reg <= 16'd0;
            active_batch <= 4'd0;
            elem_idx <= {IDX_BITS{1'b0}};
            need_kstart <= 1'b0;
            stream_done <= 1'b0;
            rd_busy_seen <= 1'b0;
            seq_start_pulse <= 1'b0;
            k_start_pulse <= 1'b0;
            k_valid_pulse <= 1'b0;
            v_valid_pulse <= 1'b0;
            v_seq_last <= 1'b0;
            score_seq_last_pulse <= 1'b0;
            dot_last_pending <= 1'b0;
            dot_last_d1 <= 1'b0;
            score_last_d1 <= 1'b0;
            rd_req <= 1'b0;
            rd_session_id <= 3'd0;
            rd_token_start <= 16'd0;
            rd_token_end <= 16'd0;
            ctx_out <= 16'd0;
            ctx_batch_id <= 3'd0;
            ctx_valid <= 1'b0;
            ctx_last <= 1'b0;
            attn_done <= 1'b0;
            attn_busy <= 1'b0;
            out_batch <= {BATCH_BITS{1'b0}};
            out_idx <= {IDX_BITS{1'b0}};
            v_wr_buf <= 1'b0;
            token_count_reg <= 16'd0;
            for (seq_i = 0; seq_i < MAX_BATCH; seq_i = seq_i + 1) begin
                ctx_done[seq_i] <= 1'b1;
                ctx_cap_idx[seq_i] <= {IDX_BITS{1'b0}};
                weight_rd_buf[seq_i] <= 1'b0;
                weight_count[seq_i] <= 16'd0;
                weight_sum[seq_i] <= 32'd0;
                for (seq_j = 0; seq_j < HEAD_DIM; seq_j = seq_j + 1) begin
                    ctx_calc[seq_i][seq_j] <= 32'd0;
                end
            end
        end else begin
            seq_start_pulse <= 1'b0;
            k_start_pulse <= 1'b0;
            k_valid_pulse <= 1'b0;
            v_valid_pulse <= 1'b0;
            v_seq_last <= 1'b0;
            ctx_valid <= 1'b0;
            ctx_last <= 1'b0;
            attn_done <= 1'b0;

            dot_last_d1 <= dot_valid_b[0] && dot_last_pending;
            score_last_d1 <= dot_last_d1;
            score_seq_last_pulse <= dot_valid_b[0] && dot_last_pending;
            if (dot_valid_b[0] && dot_last_pending) begin
                dot_last_pending <= 1'b0;
            end

            for (seq_i = 0; seq_i < MAX_BATCH; seq_i = seq_i + 1) begin
                if ((seq_i < active_batch) && ctx_valid_b[seq_i]) begin
                    ctx_buf[seq_i][ctx_cap_idx[seq_i]] <= ctx_out_b[seq_i];
                    if (ctx_cap_idx[seq_i] != HEAD_DIM[IDX_BITS-1:0] - 1'b1) begin
                        ctx_cap_idx[seq_i] <= ctx_cap_idx[seq_i] + 1'b1;
                    end
                end
                if ((seq_i < active_batch) && weight_valid_b[seq_i]) begin
                    for (seq_j = 0; seq_j < HEAD_DIM; seq_j = seq_j + 1) begin
                        ctx_calc[seq_i][seq_j] <= ctx_calc[seq_i][seq_j] +
                            ((v_buf[weight_rd_buf[seq_i]][seq_j] * weight_b[seq_i]) >> 15);
                    end
                    weight_sum[seq_i] <= weight_sum[seq_i] + weight_b[seq_i];
                    weight_rd_buf[seq_i] <= ~weight_rd_buf[seq_i];
                    weight_count[seq_i] <= weight_count[seq_i] + 1'b1;
                    if ((weight_count[seq_i] + 1'b1) >= token_count_reg) begin
                        ctx_done[seq_i] <= 1'b1;
                    end
                end
            end

            case (state)
                IDLE: begin
                    rd_req <= 1'b0;
                    attn_busy <= 1'b0;
                    stream_done <= 1'b0;
                    if (attn_start) begin
                        sess_reg <= session_id;
                        ts_reg <= token_start;
                        te_reg <= token_end;
                        active_batch <= (batch_size == 3'd0) ? 4'd8 : {1'b0, batch_size};
                        seq_start_pulse <= 1'b1;
                        elem_idx <= {IDX_BITS{1'b0}};
                        need_kstart <= 1'b1;
                        stream_done <= 1'b0;
                        rd_busy_seen <= 1'b0;
                        dot_last_pending <= 1'b0;
                        dot_last_d1 <= 1'b0;
                        score_last_d1 <= 1'b0;
                        score_seq_last_pulse <= 1'b0;
                        out_batch <= {BATCH_BITS{1'b0}};
                        out_idx <= {IDX_BITS{1'b0}};
                        v_wr_buf <= 1'b0;
                        token_count_reg <= token_end - token_start + 16'd1;
                        attn_busy <= 1'b1;
                        rd_session_id <= session_id;
                        rd_token_start <= token_start;
                        rd_token_end <= token_end;
                        for (seq_i = 0; seq_i < MAX_BATCH; seq_i = seq_i + 1) begin
                            ctx_done[seq_i] <= (seq_i >= ((batch_size == 3'd0) ? 4'd8 : {1'b0, batch_size}));
                            ctx_cap_idx[seq_i] <= {IDX_BITS{1'b0}};
                            weight_rd_buf[seq_i] <= 1'b0;
                            weight_count[seq_i] <= 16'd0;
                            weight_sum[seq_i] <= 32'd0;
                            for (seq_j = 0; seq_j < HEAD_DIM; seq_j = seq_j + 1) begin
                                ctx_calc[seq_i][seq_j] <= 32'd0;
                            end
                        end
                        state <= FETCH_KV;
                    end
                end

                FETCH_KV: begin
                    attn_busy <= 1'b1;
                    rd_req <= 1'b1;
                    rd_session_id <= sess_reg;
                    rd_token_start <= ts_reg;
                    rd_token_end <= te_reg;
                    if (rd_busy) begin
                        rd_busy_seen <= 1'b1;
                    end
                    if (rd_busy_seen && rd_busy) begin
                        state <= K_START;
                    end
                end

                K_START: begin
                    attn_busy <= 1'b1;
                    rd_req <= 1'b0;
                    k_start_pulse <= 1'b1;
                    need_kstart <= 1'b0;
                    state <= STREAM;
                end

                STREAM: begin
                    attn_busy <= 1'b1;
                    rd_req <= 1'b0;

                    if (!stream_done && !any_stall) begin
                        if (need_kstart) begin
                            k_start_pulse <= 1'b1;
                            need_kstart <= 1'b0;
                            if (rd_valid) begin
                                k_valid_pulse <= 1'b1;
                                v_valid_pulse <= 1'b1;
                                v_buf[v_wr_buf][elem_idx] <= rd_v_data;
                                if (rd_last) begin
                                    v_seq_last <= 1'b1;
                                    stream_done <= 1'b1;
                                    dot_last_pending <= 1'b1;
                                end
                                if (elem_idx == HEAD_DIM[IDX_BITS-1:0] - 1'b1) begin
                                    elem_idx <= {IDX_BITS{1'b0}};
                                    v_wr_buf <= ~v_wr_buf;
                                    if (!rd_last) begin
                                        need_kstart <= 1'b1;
                                    end
                                end else begin
                                    elem_idx <= elem_idx + 1'b1;
                                end
                            end
                        end else if (rd_valid) begin
                            k_valid_pulse <= 1'b1;
                            v_valid_pulse <= 1'b1;
                            v_buf[v_wr_buf][elem_idx] <= rd_v_data;
                            if (rd_last) begin
                                v_seq_last <= 1'b1;
                                stream_done <= 1'b1;
                                dot_last_pending <= 1'b1;
                            end
                            if (elem_idx == HEAD_DIM[IDX_BITS-1:0] - 1'b1) begin
                                elem_idx <= {IDX_BITS{1'b0}};
                                v_wr_buf <= ~v_wr_buf;
                                if (!rd_last) begin
                                    need_kstart <= 1'b1;
                                end
                            end else begin
                                elem_idx <= elem_idx + 1'b1;
                            end
                        end
                    end

                    if (stream_done && all_done) begin
                        out_batch <= {BATCH_BITS{1'b0}};
                        out_idx <= {IDX_BITS{1'b0}};
                        state <= OUTPUT;
                    end
                end

                OUTPUT: begin
                    attn_busy <= 1'b1;
                    rd_req <= 1'b0;
                    ctx_out <= norm_calc[15:0];
                    ctx_batch_id <= out_batch;
                    ctx_valid <= 1'b1;
                    ctx_last <= ((out_batch == active_batch[BATCH_BITS-1:0] - 1'b1) &&
                                 (out_idx == HEAD_DIM[IDX_BITS-1:0] - 1'b1));
                    if ((out_batch == active_batch[BATCH_BITS-1:0] - 1'b1) &&
                        (out_idx == HEAD_DIM[IDX_BITS-1:0] - 1'b1)) begin
                        attn_done <= 1'b1;
                        state <= IDLE;
                    end else if (out_idx == HEAD_DIM[IDX_BITS-1:0] - 1'b1) begin
                        out_idx <= {IDX_BITS{1'b0}};
                        out_batch <= out_batch + 1'b1;
                    end else begin
                        out_idx <= out_idx + 1'b1;
                    end
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
