`timescale 1ns/1ps
// -----------------------------------------------------------------------
// Module : tb_attention_ctrl
// Project: Kael -- Attention Accelerator (Circle Inference Silicon IP)
// Purpose: Integration testbench, direct streaming architecture
// -----------------------------------------------------------------------

module tb_attention_ctrl;

    logic clk;
    logic rst_n;
    logic [15:0] q_data;
    logic [5:0]  q_addr;
    logic [2:0]  q_batch_id;
    logic        q_valid;
    logic [2:0]  batch_size;
    logic [2:0]  session_id;
    logic [15:0] token_start;
    logic [15:0] token_end;
    logic        attn_start;
    logic        rd_req;
    logic [2:0]  rd_session_id;
    logic [15:0] rd_token_start;
    logic [15:0] rd_token_end;
    logic [15:0] rd_k_data;
    logic [15:0] rd_v_data;
    logic        rd_valid;
    logic        rd_last;
    logic        rd_busy;
    logic [15:0] ctx_out;
    logic [2:0]  ctx_batch_id;
    logic        ctx_valid;
    logic        ctx_last;
    logic        attn_done;
    logic        attn_busy;

    logic [15:0] ctx_buf_tb [0:8*64-1];
    logic [2:0]  ctx_bid_tb [0:8*64-1];
    int          ctx_count;
    int          ctx_last_count;
    logic        test_timed_out;
    int          pass_count;
    int          fail_count;
    int          watchdog_cycles;

    attention_ctrl #(
        .HEAD_DIM(64),
        .DATA_WIDTH(16),
        .NUM_SESSIONS(8),
        .MAX_BATCH(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .q_data(q_data),
        .q_addr(q_addr),
        .q_batch_id(q_batch_id),
        .q_valid(q_valid),
        .batch_size(batch_size),
        .session_id(session_id),
        .token_start(token_start),
        .token_end(token_end),
        .attn_start(attn_start),
        .rd_req(rd_req),
        .rd_session_id(rd_session_id),
        .rd_token_start(rd_token_start),
        .rd_token_end(rd_token_end),
        .rd_k_data(rd_k_data),
        .rd_v_data(rd_v_data),
        .rd_valid(rd_valid),
        .rd_last(rd_last),
        .rd_busy(rd_busy),
        .ctx_out(ctx_out),
        .ctx_batch_id(ctx_batch_id),
        .ctx_valid(ctx_valid),
        .ctx_last(ctx_last),
        .attn_done(attn_done),
        .attn_busy(attn_busy)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            watchdog_cycles <= 0;
        end else begin
            watchdog_cycles <= watchdog_cycles + 1;
            if (watchdog_cycles > 80000) begin
                $display("FAIL [watchdog]");
                $finish;
            end
        end
    end

    task automatic record_pass(input string label);
        begin
            pass_count++;
            $display("PASS [%s]", label);
        end
    endtask

    task automatic record_fail(input string label);
        begin
            fail_count++;
            $display("FAIL [%s]", label);
        end
    endtask

    task automatic pulse_start;
        begin
            @(posedge clk);
            attn_start <= 1'b1;
            @(posedge clk);
            attn_start <= 1'b0;
        end
    endtask

    task automatic load_q(input logic [2:0] bid, input logic [15:0] val);
        int i;
        begin
            @(posedge clk);
            q_batch_id <= bid;
            q_data <= val;
            q_valid <= 1'b1;
            for (i = 0; i < 64; i = i + 1) begin
                q_addr <= i[5:0];
                @(posedge clk);
            end
            q_valid <= 1'b0;
            q_addr <= 6'd0;
        end
    endtask

    task automatic vera_serve_1tok(input logic [15:0] kv);
        int i;
        begin
            wait (rd_req === 1'b1);
            @(posedge clk);
            rd_busy <= 1'b1;
            repeat (2) @(posedge clk);
            rd_busy <= 1'b0;
            @(posedge clk);
            for (i = 0; i < 64; i = i + 1) begin
                rd_k_data <= kv;
                rd_v_data <= kv;
                rd_valid <= 1'b1;
                rd_last <= (i == 63);
                @(posedge clk);
            end
            rd_valid <= 1'b0;
            rd_last <= 1'b0;
            rd_k_data <= 16'd0;
            rd_v_data <= 16'd0;
        end
    endtask

    task automatic vera_serve_2tok(input logic [15:0] k,
        input logic [15:0] v0, input logic [15:0] v1);
        int i;
        begin
            wait (rd_req === 1'b1);
            @(posedge clk);
            rd_busy <= 1'b1;
            repeat (2) @(posedge clk);
            rd_busy <= 1'b0;
            @(posedge clk);
            for (i = 0; i < 64; i = i + 1) begin
                rd_k_data <= k;
                rd_v_data <= v0;
                rd_valid <= 1'b1;
                rd_last <= 1'b0;
                @(posedge clk);
            end
            rd_valid <= 1'b0;
            rd_last <= 1'b0;
            @(posedge clk);
            for (i = 0; i < 64; i = i + 1) begin
                rd_k_data <= k;
                rd_v_data <= v1;
                rd_valid <= 1'b1;
                rd_last <= (i == 63);
                @(posedge clk);
            end
            rd_valid <= 1'b0;
            rd_last <= 1'b0;
            rd_k_data <= 16'd0;
            rd_v_data <= 16'd0;
        end
    endtask

    task automatic collect_ctx(input int expected_count);
        int cycles;
        int i;
        int kv_cnt;
        int vv_cnt;
        int dot_cnt;
        int score_cnt;
        int weight_cnt;
        int denom_cnt;
        int seq_last_cnt;
        begin
            for (i = 0; i < 8*64; i = i + 1) begin
                ctx_buf_tb[i] = 16'd0;
                ctx_bid_tb[i] = 3'd0;
            end
            ctx_count = 0;
            ctx_last_count = 0;
            test_timed_out = 1'b0;
            cycles = 0;
            kv_cnt = 0;
            vv_cnt = 0;
            dot_cnt = 0;
            score_cnt = 0;
            weight_cnt = 0;
            denom_cnt = 0;
            seq_last_cnt = 0;
            while ((attn_done !== 1'b1) && (cycles < 5000)) begin
                @(posedge clk);
                if (dut.k_valid_pulse) kv_cnt++;
                if (dut.v_valid_pulse) vv_cnt++;
                if (dut.dot_valid_b[0]) dot_cnt++;
                if (dut.score_valid_b[0]) score_cnt++;
                if (dut.weight_valid_b[0]) weight_cnt++;
                if (dut.denom_valid_b[0]) denom_cnt++;
                if (dut.score_seq_last_pulse) seq_last_cnt++;
                if (ctx_valid) begin
                    ctx_buf_tb[ctx_count] = ctx_out;
                    ctx_bid_tb[ctx_count] = ctx_batch_id;
                    ctx_count++;
                end
                if (ctx_last) begin
                    ctx_last_count++;
                end
                cycles++;
            end
            if (cycles >= 5000) begin
                test_timed_out = 1'b1;
            end
            if (ctx_count == expected_count) begin
                pass_count++;
                $display("PASS [count=%0d]", ctx_count);
            end else begin
                fail_count++;
                $display("FAIL [count exp=%0d got=%0d]", expected_count, ctx_count);
            end
        end
    endtask

    task automatic check_data(input logic [15:0] expected,
        input int tol, input string label);
        int i;
        int mismatches;
        int diff;
        begin
            mismatches = 0;
            for (i = 0; i < ctx_count; i = i + 1) begin
                diff = (ctx_buf_tb[i] > expected) ? (ctx_buf_tb[i] - expected) : (expected - ctx_buf_tb[i]);
                if (diff > tol) begin
                    mismatches++;
                end
            end
            if (mismatches == 0) begin
                record_pass(label);
            end else begin
                fail_count++;
                $display("FAIL [%s mismatches=%0d]", label, mismatches);
            end
        end
    endtask

    task automatic check_flag(input logic cond, input string label);
        begin
            if (cond) begin
                record_pass(label);
            end else begin
                record_fail(label);
            end
        end
    endtask

    task automatic check_batch_ids_t3;
        int i;
        int bad;
        begin
            bad = 0;
            for (i = 0; i < 64; i = i + 1) begin
                if (ctx_bid_tb[i] != 3'd0) begin
                    bad++;
                end
            end
            for (i = 64; i < 128; i = i + 1) begin
                if (ctx_bid_tb[i] != 3'd1) begin
                    bad++;
                end
            end
            check_flag(bad == 0, "T3_batch_ids");
        end
    endtask

    task automatic copy_ctx(input int count, output logic [15:0] dst [0:63]);
        int i;
        begin
            for (i = 0; i < count; i = i + 1) begin
                dst[i] = ctx_buf_tb[i];
            end
        end
    endtask

    task automatic check_match(input logic [15:0] ref_buf [0:63],
        input int tol, input string label);
        int i;
        int diff;
        int bad;
        begin
            bad = 0;
            for (i = 0; i < 64; i = i + 1) begin
                diff = (ctx_buf_tb[i] > ref_buf[i]) ? (ctx_buf_tb[i] - ref_buf[i]) : (ref_buf[i] - ctx_buf_tb[i]);
                if (diff > tol) begin
                    bad++;
                end
            end
            check_flag(bad == 0, label);
        end
    endtask

    logic [15:0] t1_ref [0:63];

    initial begin
        rst_n = 1'b0;
        q_data = 16'd0;
        q_addr = 6'd0;
        q_batch_id = 3'd0;
        q_valid = 1'b0;
        batch_size = 3'd0;
        session_id = 3'd0;
        token_start = 16'd0;
        token_end = 16'd0;
        attn_start = 1'b0;
        rd_k_data = 16'd0;
        rd_v_data = 16'd0;
        rd_valid = 1'b0;
        rd_last = 1'b0;
        rd_busy = 1'b0;
        pass_count = 0;
        fail_count = 0;
        test_timed_out = 1'b0;
        ctx_count = 0;
        ctx_last_count = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        load_q(3'd0, 16'h0100);
        batch_size = 3'd1;
        session_id = 3'd0;
        token_start = 16'd0;
        token_end = 16'd0;
        fork
            vera_serve_1tok(16'h0100);
        join_none
        pulse_start();
        collect_ctx(64);
        check_data(16'h0100, 3, "T1_data");
        check_flag(test_timed_out == 1'b0, "T1_timeout");
        check_flag(ctx_last_count == 1, "T1_ctx_last");
        @(posedge clk);
        check_flag(attn_busy == 1'b0, "T1_busy");
        copy_ctx(64, t1_ref);

        load_q(3'd0, 16'h0080);
        batch_size = 3'd1;
        session_id = 3'd0;
        token_start = 16'd0;
        token_end = 16'd1;
        fork
            vera_serve_2tok(16'h0080, 16'h0100, 16'h0200);
        join_none
        pulse_start();
        collect_ctx(64);
        check_data(16'h0180, 5, "T2_data");
        check_flag(test_timed_out == 1'b0, "T2_timeout");
        check_flag(ctx_last_count == 1, "T2_ctx_last");

        load_q(3'd0, 16'h0100);
        load_q(3'd1, 16'h0100);
        batch_size = 3'd2;
        session_id = 3'd0;
        token_start = 16'd0;
        token_end = 16'd0;
        fork
            vera_serve_1tok(16'h0100);
        join_none
        pulse_start();
        collect_ctx(128);
        check_data(16'h0100, 3, "T3_data");
        check_batch_ids_t3();

        load_q(3'd0, 16'h0100);
        batch_size = 3'd1;
        session_id = 3'd0;
        token_start = 16'd0;
        token_end = 16'd0;
        fork
            vera_serve_1tok(16'h0100);
        join_none
        pulse_start();
        collect_ctx(64);
        check_match(t1_ref, 2, "T4_match");

        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $finish;
    end

endmodule
