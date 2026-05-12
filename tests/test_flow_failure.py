from framework.flow import FlowBuilder
from framework.task import FunctionTask

def test_flow_continues_on_failure():
    """Test that flow continues to independent tasks when stop_on_failure=False."""
    print("Testing flow with stop_on_failure=False...")
    
    # Track which tasks ran
    ran = []
    
    def task_success(ctx):
        # We'll just append a generic success marker since task_name isn't in ctx
        ran.append("success")
        return "ok"
        
    def task_fail(ctx):
        ran.append("fail_task")
        raise ValueError("Intentional Failure")

    # Build flow: 
    # start -> fail_task
    # start -> continue_task
    flow = (FlowBuilder("ResilientFlow")
            .stop_on_failure(False)
            .task(FunctionTask("start", task_success, max_retries=0))
            .task(FunctionTask("fail_task", task_fail, max_retries=0))
            .task(FunctionTask("continue_task", task_success, max_retries=0))

            .dependency("fail_task", "start")
            .dependency("continue_task", "start")
            .build())

    result = flow.execute(parallel=False)
    
    print(f"Tasks that ran: {ran}")
    assert ran.count("success") == 2, "Both success tasks should have run"
    assert "fail_task" in ran
    assert result.success is False
    print("Test PASSED: Flow continued after failure.")

def test_flow_stops_on_failure_default():
    """Test that flow stops on failure by default."""
    print("\nTesting default behavior (stop_on_failure=True)...")
    
    ran = []
    def task_success(ctx):
        ran.append("start")
        return "ok"
    def task_fail(ctx):
        ran.append("fail")
        raise ValueError("Fail")
    def task_should_not_run(ctx):
        ran.append("should_not_run")
        return "bad"

    flow = (FlowBuilder("StrictFlow")
            .task(FunctionTask("start", task_success, max_retries=0))
            .task(FunctionTask("fail", task_fail, max_retries=0))
            .task(FunctionTask("end", task_should_not_run, max_retries=0))
            .chain("start", "fail", "end")
            .build())


    result = flow.execute(parallel=False)
    
    print(f"Tasks that ran: {ran}")
    assert "should_not_run" not in ran
    assert "fail" in ran
    print("Test PASSED: Flow stopped on failure as expected.")

if __name__ == "__main__":
    test_flow_stops_on_failure_default()
    test_flow_continues_on_failure()
    print("\nPR #5 verification successful!")
