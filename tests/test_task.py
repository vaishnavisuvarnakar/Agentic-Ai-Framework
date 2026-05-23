import time
from framework.task import FunctionTask

def test_task_timeout_enforcement():
    # A function that sleeps for 2 seconds
    def slow_function(ctx):
        time.sleep(2.0)
        return "finished"
    
    # Create a task with a 0.5s timeout and 0 retries
    task = FunctionTask(
        name="slow_task",
        func=slow_function,
        timeout=0.5,
        max_retries=0
    )
    
    # Execute the task
    result = task.execute({})
    
    # Verify that the task failed due to timeout
    assert result.success is False
    assert result.error is not None
    assert "timed out after 0.5s" in result.error

def test_task_completes_within_timeout():
    # A function that sleeps for 0.1 seconds
    def fast_function(ctx):
        time.sleep(0.1)
        return "success"
    
    # Create a task with a 0.5s timeout
    task = FunctionTask(
        name="fast_task",
        func=fast_function,
        timeout=0.5
    )
    
    result = task.execute({})
    
    assert result.success is True
    assert result.output == "success"

if __name__ == "__main__":
    print("Running task timeout test...")
    test_task_timeout_enforcement()
    print("test_task_timeout_enforcement PASSED")
    
    print("Running fast task test...")
    test_task_completes_within_timeout()
    print("test_task_completes_within_timeout PASSED")
    
    print("\nAll tests passed successfully!")
