import time
from framework.memory import InMemoryBackend, FileBackend, MemoryEntry

def test_in_memory_ttl_zero():
    print("Testing InMemoryBackend with ttl=0...")
    backend = InMemoryBackend()
    
    # 1. Set a value with a long TTL
    backend.set("test_key", "initial_value", ttl=100)
    entry = backend.get("test_key")
    assert entry.value == "initial_value"
    assert entry.ttl == 100
    
    # 2. Update the value with ttl=0 (should expire immediately)
    backend.set("test_key", "new_value", ttl=0)
    
    # In the current implementation, get() checks for expiration
    entry_after = backend.get("test_key")
    assert entry_after is None, "Entry should have expired immediately with ttl=0"
    print("InMemoryBackend test PASSED")

def test_file_backend_ttl_zero():
    print("\nTesting FileBackend with ttl=0...")
    import os
    test_file = "test_memory.json"
    if os.path.exists(test_file):
        os.remove(test_file)
        
    try:
        backend = FileBackend(test_file)
        
        # 1. Set a value with a long TTL
        backend.set("test_key", "initial_value", ttl=100)
        entry = backend.get("test_key")
        assert entry.value == "initial_value"
        
        # 2. Update the value with ttl=0
        backend.set("test_key", "new_value", ttl=0)
        
        entry_after = backend.get("test_key")
        assert entry_after is None, "FileEntry should have expired immediately with ttl=0"
        print("FileBackend test PASSED")
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_in_memory_ttl_zero()
    test_file_backend_ttl_zero()
    print("\nAll PR #2 tests passed!")
