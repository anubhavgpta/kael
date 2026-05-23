// -----------------------------------------------------------------------------
// Module : tb_v_accumulator
// Project: Kael -- Attention Accelerator (Circle Inference Silicon IP)
// Purpose: xsim testbench for v_accumulator.v
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module tb_v_accumulator;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam HEAD_DIM     = 64;
    localparam DATA_WIDTH   = 16;
    localparam WEIGHT_WIDTH = 16;
    localparam ACC_WIDTH    = 32;

    // -------------------------------------------------------------------------
    // DUT ports
    // -------------------------------------------------------------------------
    logic                     clk;
    logic                     rst_n;
    logic [DATA_WIDTH-1:0]    v_data;
    logic                     v_valid;
    logic [WEIGHT_WIDTH-1:0]  weight_in;
    logic                     weight_valid;
    logic                     rescale_valid;
    logic [WEIGHT_WIDTH-1:0]  rescale_factor;
    logic                     seq_start;
    logic                     seq_last;
    logic [ACC_WIDTH-1:0]     denom_in;
    logic                     denom_valid;
    logic [DATA_WIDTH-1:0]    ctx_out;
    logic                     ctx_valid;
    logic                     ctx_last;
    logic                     stall;

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    v_accumulator #(
        .HEAD_DIM    (HEAD_DIM),
        .DATA_WIDTH  (DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .v_data        (v_data),
        .v_valid       (v_valid),
        .weight_in     (weight_in),
        .weight_valid  (weight_valid),
        .rescale_valid (rescale_valid),
        .rescale_factor(rescale_factor),
        .seq_start     (seq_start),
        .seq_last      (seq_last),
        .denom_in      (denom_in),
        .denom_valid   (denom_valid),
        .ctx_out       (ctx_out),
        .ctx_valid     (ctx_valid),
        .ctx_last      (ctx_last),
        .stall         (stall)
    );

    // -------------------------------------------------------------------------
    // Clock: 10ns period (100 MHz)
    // -------------------------------------------------------------------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // PASS/FAIL counters
    // -------------------------------------------------------------------------
    integer pass_cnt;
    integer fail_cnt;

    // -------------------------------------------------------------------------
    // Task: pass/fail helper
    // -------------------------------------------------------------------------
    task automatic check_cond(
        input bit    cond,
        input string label
    );
        if (cond) begin
            $display("PASS [%s]", label);
            pass_cnt++;
        end else begin
            $display("FAIL [%s]", label);
            fail_cnt++;
        end
    endtask

    // -------------------------------------------------------------------------
    // Function: signed +/-1 LSB tolerance for Q8.8 output checks
    // -------------------------------------------------------------------------
    function automatic bit within_1_lsb(
        input logic [DATA_WIDTH-1:0] got,
        input logic [DATA_WIDTH-1:0] expected
    );
        integer got_i;
        integer exp_i;
        integer diff;
        begin
            got_i = $signed(got);
            exp_i = $signed(expected);
            diff  = got_i - exp_i;
            if (diff < 0)
                diff = -diff;
            within_1_lsb = (diff <= 1);
        end
    endfunction

    // -------------------------------------------------------------------------
    // Task: pulse_seq_start
    // -------------------------------------------------------------------------
    task automatic pulse_seq_start;
        @(negedge clk);
        seq_start = 1'b1;
        @(negedge clk);
        seq_start = 1'b0;
    endtask

    // -------------------------------------------------------------------------
    // Task: send_token
    //   Pulses weight_valid for one cycle, then streams HEAD_DIM V elements.
    //   weight_in remains stable for the whole V vector.
    // -------------------------------------------------------------------------
    task automatic send_token(
        input logic [WEIGHT_WIDTH-1:0] weight,
        input logic [DATA_WIDTH-1:0]   v_val,
        input logic                    is_last
    );
        integer i;
        begin
            @(negedge clk);
            weight_in    = weight;
            weight_valid = 1'b1;
            v_valid      = 1'b0;
            seq_last     = 1'b0;

            @(negedge clk);
            weight_valid = 1'b0;

            for (i = 0; i < HEAD_DIM; i = i + 1) begin
                v_data   = v_val;
                v_valid  = 1'b1;
                seq_last = is_last && (i == (HEAD_DIM - 1));
                @(negedge clk);
            end

            v_valid  = 1'b0;
            seq_last = 1'b0;
            v_data   = '0;
        end
    endtask

    // -------------------------------------------------------------------------
    // Task: send_denom
    // -------------------------------------------------------------------------
    task automatic send_denom(
        input logic [ACC_WIDTH-1:0] denom
    );
        begin
            @(negedge clk);
            denom_in    = denom;
            denom_valid = 1'b1;
            @(negedge clk);
            denom_valid = 1'b0;
        end
    endtask

    // -------------------------------------------------------------------------
    // Task: collect_and_check_context
    //   Waits for the output stream and checks all HEAD_DIM ctx_valid beats.
    // -------------------------------------------------------------------------
    task automatic collect_and_check_context(
        input logic [DATA_WIDTH-1:0] expected,
        input string                 label
    );
        integer timeout;
        integer count;
        integer last_count;
        bit     mismatch;
        bit     done;
        begin
            timeout    = 0;
            count      = 0;
            last_count = 0;
            mismatch   = 1'b0;
            done       = 1'b0;

            while ((timeout < 256) && !done) begin
                @(posedge clk);
                #1;
                timeout++;

                if (ctx_valid) begin
                    count++;
                    if (!within_1_lsb(ctx_out, expected)) begin
                        $display("FAIL [%s:data%0d]: expected 0x%04X got 0x%04X",
                                 label, count - 1, expected, ctx_out);
                        fail_cnt++;
                        mismatch = 1'b1;
                    end

                    if (ctx_last) begin
                        last_count++;
                        if (count != HEAD_DIM) begin
                            $display("FAIL [%s:ctx_last]: asserted on element %0d",
                                     label, count - 1);
                            fail_cnt++;
                        end
                    end

                    if (count == HEAD_DIM)
                        done = 1'b1;
                end else if (ctx_last) begin
                    $display("FAIL [%s:ctx_last]: asserted without ctx_valid", label);
                    fail_cnt++;
                end
            end

            if (count != HEAD_DIM) begin
                $display("FAIL [%s:ctx_valid_count]: expected 64 got %0d", label, count);
                fail_cnt++;
            end else begin
                @(posedge clk);
                #1;
                if (ctx_valid) begin
                    $display("FAIL [%s:ctx_valid_count]: extra ctx_valid after 64 pulses",
                             label);
                    fail_cnt++;
                end else begin
                    $display("PASS [%s:ctx_valid_count]: exactly 64 pulses", label);
                    pass_cnt++;
                end
            end

            if (last_count != 1) begin
                $display("FAIL [%s:ctx_last_count]: expected 1 got %0d", label, last_count);
                fail_cnt++;
            end else begin
                $display("PASS [%s:ctx_last_count]: one pulse on element 63", label);
                pass_cnt++;
            end

            if (!mismatch && (count == HEAD_DIM)) begin
                $display("PASS [%s:data]: all ctx_out values near 0x%04X",
                         label, expected);
                pass_cnt++;
            end
        end
    endtask

    // -------------------------------------------------------------------------
    // Task: apply_rescale_and_check_stall
    // -------------------------------------------------------------------------
    task automatic apply_rescale_and_check_stall(
        input logic [WEIGHT_WIDTH-1:0] factor,
        input string                   label
    );
        integer i;
        bit stall_ok;
        begin
            stall_ok = 1'b1;

            @(negedge clk);
            rescale_factor = factor;
            rescale_valid  = 1'b1;

            @(posedge clk);
            #1;
            if (!stall)
                stall_ok = 1'b0;

            @(negedge clk);
            rescale_valid = 1'b0;

            for (i = 1; i < HEAD_DIM; i = i + 1) begin
                @(posedge clk);
                #1;
                if (!stall)
                    stall_ok = 1'b0;
            end

            @(posedge clk);
            #1;
            check_cond(stall_ok, {label, ":stall asserted for 64 cycles"});
            check_cond(!stall, {label, ":stall deasserted after rescale"});
        end
    endtask

    // -------------------------------------------------------------------------
    // Main stimulus
    // -------------------------------------------------------------------------
    initial begin
        rst_n          = 1'b0;
        v_data         = '0;
        v_valid        = 1'b0;
        weight_in      = '0;
        weight_valid   = 1'b0;
        rescale_valid  = 1'b0;
        rescale_factor = '0;
        seq_start      = 1'b0;
        seq_last       = 1'b0;
        denom_in       = '0;
        denom_valid    = 1'b0;
        pass_cnt       = 0;
        fail_cnt       = 0;

        repeat(4) @(negedge clk);
        rst_n = 1'b1;
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // Test 1: Single token, uniform V
        // -----------------------------------------------------------------
        $display("TEST 1: single token, uniform V");
        pulse_seq_start();
        send_token(16'h7FFF, 16'h0100, 1'b1);
        send_denom(32'h0000_7FFF);
        collect_and_check_context(16'h0100, "T1");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // Test 2: Two tokens, equal weights
        // -----------------------------------------------------------------
        $display("TEST 2: two tokens, equal weights");
        pulse_seq_start();
        send_token(16'h4000, 16'h0200, 1'b0);
        send_token(16'h4000, 16'h0200, 1'b1);
        send_denom(32'h0000_7FFF);
        collect_and_check_context(16'h0200, "T2");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // Test 3: Rescale fires mid-sequence
        // -----------------------------------------------------------------
        $display("TEST 3: rescale fires mid-sequence");
        pulse_seq_start();
        send_token(16'h7FFF, 16'h0100, 1'b0);
        apply_rescale_and_check_stall(16'h4000, "T3");
        send_token(16'h7FFF, 16'h0100, 1'b1);
        send_denom(32'h0000_7FFF);
        collect_and_check_context(16'h0180, "T3");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // Test 4: ctx_last timing
        // -----------------------------------------------------------------
        $display("TEST 4: ctx_last timing");
        pulse_seq_start();
        send_token(16'h7FFF, 16'h0100, 1'b1);
        send_denom(32'h0000_7FFF);
        collect_and_check_context(16'h0100, "T4");
        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // Test 5: seq_start clears accumulator
        // -----------------------------------------------------------------
        $display("TEST 5: seq_start clears accumulator");
        pulse_seq_start();
        send_token(16'h7FFF, 16'h0100, 1'b1);
        send_denom(32'h0000_7FFF);
        collect_and_check_context(16'h0100, "T5a");

        pulse_seq_start();
        send_token(16'h0000, 16'h0100, 1'b1);
        send_denom(32'h0000_7FFF);
        collect_and_check_context(16'h0000, "T5b");

        repeat(2) @(negedge clk);

        // -----------------------------------------------------------------
        // Summary
        // -----------------------------------------------------------------
        $display("--------------------------------------------------");
        $display("Results: %0d PASS, %0d FAIL", pass_cnt, fail_cnt);
        $display("--------------------------------------------------");
        $finish;
    end

    // -------------------------------------------------------------------------
    // Watchdog: abort if simulation hangs
    // -------------------------------------------------------------------------
    initial begin
        #200000;
        $display("FAIL [watchdog]: simulation exceeded time limit");
        $finish;
    end

endmodule
