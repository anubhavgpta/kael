// -----------------------------------------------------------------------------
// Module : tb_qk_dot_engine
// Project: Kael -- Attention Accelerator (Circle Inference Silicon IP)
// Purpose: xsim testbench for qk_dot_engine.v
// -----------------------------------------------------------------------------
//
// Tests
// -----
// 1. Known dot product: Q=[1,2,3,4,0..0], K=[1,2,3,4,0..0]  -> 30.0
// 2. All-ones:          Q=[1.0 x64],       K=[1.0 x64]       -> 64.0
// 3. Batch isolation:   batch0=2.0, batch1=3.0, K=1.0 x64
//                       batch0 -> 128.0, batch1 -> 192.0
// 4. k_start protocol:  dot_valid arrives exactly 65 cycles after first k_valid
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module tb_qk_dot_engine;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam HEAD_DIM   = 64;
    localparam DATA_WIDTH = 16;
    localparam PE_COUNT   = 8;
    localparam ACC_WIDTH  = 32;
    localparam MAX_BATCH  = 8;

    // Q16.16 fixed-point constants
    localparam logic [ACC_WIDTH-1:0] Q16_30  = 32'h001E_0000; // 30.0
    localparam logic [ACC_WIDTH-1:0] Q16_64  = 32'h0040_0000; // 64.0
    localparam logic [ACC_WIDTH-1:0] Q16_128 = 32'h0080_0000; // 128.0
    localparam logic [ACC_WIDTH-1:0] Q16_192 = 32'h00C0_0000; // 192.0

    // Q8.8 value for 1.0 and 2.0 and 3.0
    localparam logic [DATA_WIDTH-1:0] Q88_1 = 16'h0100; // 1.0
    localparam logic [DATA_WIDTH-1:0] Q88_2 = 16'h0200; // 2.0
    localparam logic [DATA_WIDTH-1:0] Q88_3 = 16'h0300; // 3.0
    localparam logic [DATA_WIDTH-1:0] Q88_4 = 16'h0400; // 4.0

    // -------------------------------------------------------------------------
    // DUT ports
    // -------------------------------------------------------------------------
    logic                          clk;
    logic                          rst_n;

    logic [DATA_WIDTH-1:0]         q_data;
    logic                          q_valid;
    logic [$clog2(HEAD_DIM)-1:0]   q_addr;
    logic [$clog2(MAX_BATCH)-1:0]  q_batch_id;

    logic [DATA_WIDTH-1:0]         k_data;
    logic                          k_valid;
    logic                          k_start;
    logic [$clog2(MAX_BATCH)-1:0]  k_batch_id;

    logic [ACC_WIDTH-1:0]          dot_result;
    logic                          dot_valid;

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    qk_dot_engine #(
        .HEAD_DIM  (HEAD_DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .PE_COUNT  (PE_COUNT),
        .ACC_WIDTH (ACC_WIDTH),
        .MAX_BATCH (MAX_BATCH)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .q_data    (q_data),
        .q_valid   (q_valid),
        .q_addr    (q_addr),
        .q_batch_id(q_batch_id),
        .k_data    (k_data),
        .k_valid   (k_valid),
        .k_start   (k_start),
        .k_batch_id(k_batch_id),
        .dot_result(dot_result),
        .dot_valid (dot_valid)
    );

    // -------------------------------------------------------------------------
    // Clock generation: 10ns period (100 MHz)
    // -------------------------------------------------------------------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // PASS/FAIL counters
    // -------------------------------------------------------------------------
    integer pass_cnt;
    integer fail_cnt;

    // -------------------------------------------------------------------------
    // Task: load_q
    //   Loads all 64 elements of a Q vector into a given batch bank.
    //   One element per cycle, q_valid held high for 64 consecutive cycles.
    // -------------------------------------------------------------------------
    task automatic load_q(
        input logic [$clog2(MAX_BATCH)-1:0] bid,
        input logic [DATA_WIDTH-1:0]        qvec [HEAD_DIM]
    );
        integer i;
        for (i = 0; i < HEAD_DIM; i++) begin
            @(negedge clk);
            q_batch_id = bid;
            q_addr     = i[$clog2(HEAD_DIM)-1:0];
            q_data     = qvec[i];
            q_valid    = 1'b1;
        end
        @(negedge clk);
        q_valid = 1'b0;
    endtask

    // -------------------------------------------------------------------------
    // Task: stream_k
    //   Pulses k_start for one cycle, waits one cycle (protocol: k_valid must
    //   not be asserted on the same cycle as k_start), then streams 64 k_valid
    //   cycles with consecutive K elements.
    // -------------------------------------------------------------------------
    task automatic stream_k(
        input logic [$clog2(MAX_BATCH)-1:0] bid,
        input logic [DATA_WIDTH-1:0]        kvec [HEAD_DIM]
    );
        integer i;
        // k_start pulse
        @(negedge clk);
        k_batch_id = bid;
        k_start    = 1'b1;
        k_valid    = 1'b0;
        @(negedge clk);
        k_start    = 1'b0;           // deassert k_start
        // mandatory one-cycle gap before first k_valid
        k_valid    = 1'b0;
        // stream K elements
        for (i = 0; i < HEAD_DIM; i++) begin
            @(negedge clk);
            k_data  = kvec[i];
            k_valid = 1'b1;
        end
        @(negedge clk);
        k_valid = 1'b0;
    endtask

    // -------------------------------------------------------------------------
    // Task: wait_dot
    //   Waits for dot_valid (timeout 200 cycles), then checks result.
    //   Logs PASS or FAIL with the provided label.
    // -------------------------------------------------------------------------
    task automatic wait_dot(
        input logic [ACC_WIDTH-1:0] expected,
        input string                label
    );
        integer timeout;
        timeout = 0;
        while (!dot_valid && timeout < 200) begin
            @(posedge clk);
            timeout++;
        end
        if (!dot_valid) begin
            $display("FAIL [%s]: dot_valid never asserted (timeout)", label);
            fail_cnt++;
        end else if (dot_result !== expected) begin
            $display("FAIL [%s]: expected 0x%08X got 0x%08X",
                     label, expected, dot_result);
            fail_cnt++;
        end else begin
            $display("PASS [%s]: dot_result=0x%08X", label, dot_result);
            pass_cnt++;
        end
    endtask

    // -------------------------------------------------------------------------
    // Test vectors
    // -------------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] q_vec [HEAD_DIM];
    logic [DATA_WIDTH-1:0] k_vec [HEAD_DIM];

    // -------------------------------------------------------------------------
    // Main stimulus
    // -------------------------------------------------------------------------
    initial begin
        // Default idle values
        rst_n      = 1'b0;
        q_data     = '0;
        q_valid    = 1'b0;
        q_addr     = '0;
        q_batch_id = '0;
        k_data     = '0;
        k_valid    = 1'b0;
        k_start    = 1'b0;
        k_batch_id = '0;
        pass_cnt   = 0;
        fail_cnt   = 0;

        // Release reset after 4 cycles
        repeat(4) @(negedge clk);
        rst_n = 1'b1;
        repeat(2) @(negedge clk);

        // =================================================================
        // Test 1: Known dot product
        //   Q = [1.0, 2.0, 3.0, 4.0, 0.0 x 60]
        //   K = [1.0, 2.0, 3.0, 4.0, 0.0 x 60]
        //   Expected: 1^2 + 2^2 + 3^2 + 4^2 = 30.0 -> Q16.16 = 0x001E0000
        // =================================================================
        begin : test1
            integer i;
            for (i = 0; i < HEAD_DIM; i++) q_vec[i] = '0;
            for (i = 0; i < HEAD_DIM; i++) k_vec[i] = '0;
            q_vec[0] = Q88_1; q_vec[1] = Q88_2;
            q_vec[2] = Q88_3; q_vec[3] = Q88_4;
            k_vec[0] = Q88_1; k_vec[1] = Q88_2;
            k_vec[2] = Q88_3; k_vec[3] = Q88_4;

            load_q(3'h0, q_vec);
            stream_k(3'h0, k_vec);
            wait_dot(Q16_30, "T1:known_dot");
        end

        repeat(4) @(negedge clk);

        // =================================================================
        // Test 2: All-ones dot product
        //   Q = [1.0 x 64], K = [1.0 x 64]
        //   Expected: 64.0 -> Q16.16 = 0x00400000
        // =================================================================
        begin : test2
            integer i;
            for (i = 0; i < HEAD_DIM; i++) q_vec[i] = Q88_1;
            for (i = 0; i < HEAD_DIM; i++) k_vec[i] = Q88_1;

            load_q(3'h1, q_vec);
            stream_k(3'h1, k_vec);
            wait_dot(Q16_64, "T2:all_ones");
        end

        repeat(4) @(negedge clk);

        // =================================================================
        // Test 3: Batch isolation
        //   batch 0: Q = [2.0 x 64]
        //   batch 1: Q = [3.0 x 64]
        //   K = [1.0 x 64]
        //   batch 0: 64 * 2.0 * 1.0 = 128.0  -> 0x00800000
        //   batch 1: 64 * 3.0 * 1.0 = 192.0  -> 0x00C00000
        // =================================================================
        begin : test3
            integer i;
            for (i = 0; i < HEAD_DIM; i++) q_vec[i] = Q88_2;
            load_q(3'h0, q_vec);
            for (i = 0; i < HEAD_DIM; i++) q_vec[i] = Q88_3;
            load_q(3'h1, q_vec);
            for (i = 0; i < HEAD_DIM; i++) k_vec[i] = Q88_1;

            stream_k(3'h0, k_vec);
            wait_dot(Q16_128, "T3a:batch0");

            repeat(4) @(negedge clk);

            for (i = 0; i < HEAD_DIM; i++) k_vec[i] = Q88_1;
            stream_k(3'h1, k_vec);
            wait_dot(Q16_192, "T3b:batch1");
        end

        repeat(4) @(negedge clk);

        // =================================================================
        // Test 4: k_start protocol -- dot_valid exactly 65 cycles after
        //         the first k_valid pulse.
        //   Use all-ones again (result is deterministic and easy to verify).
        //   Measure cycle count from first k_valid to dot_valid assertion.
        // =================================================================
        begin : test4
            integer i;
            integer cycle_count;
            integer first_kv_time;
            integer dv_time;
            logic   found;

            for (i = 0; i < HEAD_DIM; i++) q_vec[i] = Q88_1;
            for (i = 0; i < HEAD_DIM; i++) k_vec[i] = Q88_1;

            load_q(3'h2, q_vec);

            // k_start pulse
            @(negedge clk);
            k_batch_id = 3'h2;
            k_start    = 1'b1;
            k_valid    = 1'b0;
            @(negedge clk);
            k_start    = 1'b0;
            k_valid    = 1'b0;

            // Count rising edges from first k_valid to dot_valid
            cycle_count    = 0;
            first_kv_time  = 0;
            dv_time        = 0;
            found          = 1'b0;

            // Stream first k_valid; start counting from this edge
            @(negedge clk);
            k_data  = k_vec[0];
            k_valid = 1'b1;

            // Count this cycle as cycle 1
            @(posedge clk); // cycle 1 edge -- first k_valid captured
            cycle_count = 1;

            // Stream remaining 63 elements and keep counting
            for (i = 1; i < HEAD_DIM; i++) begin
                @(negedge clk);
                k_data  = k_vec[i];
                k_valid = 1'b1;
                @(posedge clk);
                cycle_count++;
                if (dot_valid && !found) begin
                    dv_time = cycle_count;
                    found   = 1'b1;
                end
            end

            // k_valid off; wait a few more cycles for dot_valid
            @(negedge clk);
            k_valid = 1'b0;

            // Scan up to 5 more posedges.
            // #1 advances past the NBA region so always_ff outputs are visible.
            begin : scan_loop
                integer s;
                for (s = 0; s < 5 && !found; s++) begin
                    @(posedge clk);
                    #1;
                    cycle_count++;
                    if (dot_valid) begin
                        dv_time = cycle_count;
                        found   = 1'b1;
                    end
                end
            end

            if (!found) begin
                $display("FAIL [T4:timing]: dot_valid never asserted");
                fail_cnt++;
            end else if (dv_time !== 65) begin
                $display("FAIL [T4:timing]: expected dot_valid on cycle 65, got cycle %0d",
                         dv_time);
                fail_cnt++;
            end else begin
                $display("PASS [T4:timing]: dot_valid on cycle %0d (correct)", dv_time);
                pass_cnt++;
            end

            // Also confirm result value
            if (dot_valid) begin
                if (dot_result !== Q16_64) begin
                    $display("FAIL [T4:result]: expected 0x%08X got 0x%08X",
                             Q16_64, dot_result);
                    fail_cnt++;
                end else begin
                    $display("PASS [T4:result]: dot_result=0x%08X", dot_result);
                    pass_cnt++;
                end
            end
        end

        repeat(4) @(negedge clk);

        // =================================================================
        // Summary
        // =================================================================
        $display("--------------------------------------------------");
        $display("Results: %0d PASS, %0d FAIL", pass_cnt, fail_cnt);
        $display("--------------------------------------------------");
        $finish;
    end

    // -------------------------------------------------------------------------
    // Watchdog: kill simulation if it runs longer than expected
    // -------------------------------------------------------------------------
    initial begin
        #200000;
        $display("FAIL [watchdog]: simulation exceeded time limit");
        $finish;
    end

endmodule
