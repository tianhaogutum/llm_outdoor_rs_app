#!/usr/bin/env python3
"""
Ollama Diagnosis Script - Check if Ollama service is working properly
Usage: python backend/scripts/diagnose_ollama.py
"""
import httpx
import sys
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.settings import get_settings

def check_ollama_service():
    """Check Ollama service status"""
    print("=" * 60)
    print("🔍 Ollama Diagnosis Tool")
    print("=" * 60)
    
    settings = get_settings()
    # Normalize to use 127.0.0.1 instead of localhost to avoid IPv6 issues
    base_url = settings.ollama_api_url.replace("/api/generate", "").replace("localhost", "127.0.0.1")
    api_url = settings.ollama_api_url.replace("localhost", "127.0.0.1")
    
    print(f"\n📋 Configuration Info:")
    print(f"   API URL: {settings.ollama_api_url}")
    print(f"   Model: {settings.ollama_model}")
    print(f"   Timeout: {settings.ollama_timeout}s")
    print(f"   Base URL: {base_url}")
    
    issues = []
    
    # Step 1: Check if service is running
    print(f"\n[1/4] Checking if Ollama service is running...")
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            print("   ✅ Ollama service is running")
            
            # Display installed models
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            print(f"   📦 Installed models: {', '.join(model_names) if model_names else 'None'}")
            
    except httpx.ConnectError:
        print("   ❌ Ollama service is not running")
        print("   💡 Try running: brew services start ollama")
        issues.append("Ollama service is not running")
        return False, issues
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        issues.append(f"Connection failed: {e}")
        return False, issues
    
    # Step 2: Check if model exists
    print(f"\n[2/4] Checking if model '{settings.ollama_model}' is available...")
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/api/tags")
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if settings.ollama_model in model_names:
                print(f"   ✅ Model '{settings.ollama_model}' is installed")
            else:
                print(f"   ❌ Model '{settings.ollama_model}' is not installed")
                print(f"   💡 Run: ollama pull {settings.ollama_model}")
                issues.append(f"Model '{settings.ollama_model}' is not installed")
                return False, issues
    except Exception as e:
        print(f"   ❌ Check failed: {e}")
        issues.append(f"Model check failed: {e}")
        return False, issues
    
    # Step 3: Test simple generation
    print(f"\n[3/4] Testing model generation...")
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                api_url,
                json={
                    "model": settings.ollama_model,
                    "prompt": "Say 'Hello' in one word",
                    "stream": False,
                }
            )
            response.raise_for_status()
            result = response.json()
            
            if "response" in result:
                generated_text = result["response"].strip()
                print(f"   ✅ Generation successful")
                print(f"   📝 Response: {generated_text[:50]}...")
            else:
                print(f"   ⚠️  Abnormal response format")
                issues.append("Abnormal response format")
                return False, issues
    except httpx.HTTPStatusError as e:
        print(f"   ❌ HTTP Error: {e.response.status_code}")
        error_text = e.response.text[:200]
        print(f"   📄 Response: {error_text}")
        issues.append(f"HTTP {e.response.status_code}: {error_text}")
        return False, issues
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        issues.append(f"Generation failed: {e}")
        return False, issues
    
    # Step 4: Test complete flow (simulate backend call)
    print(f"\n[4/4] Testing complete API call...")
    try:
        with httpx.Client(timeout=settings.ollama_timeout) as client:
            response = client.post(
                api_url,
                json={
                    "model": settings.ollama_model,
                    "prompt": "Test prompt",
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "num_predict": 10,
                    },
                }
            )
            response.raise_for_status()
            result = response.json()
            
            if "response" in result and result.get("done", False):
                print("   ✅ Complete flow test successful")
            else:
                print("   ⚠️  Incomplete response")
                issues.append("Incomplete response")
                return False, issues
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        issues.append(f"Complete test failed: {e}")
        return False, issues
    
    print("\n" + "=" * 60)
    print("✅ All checks passed! Ollama configuration is normal.")
    print("=" * 60)
    return True, []

if __name__ == "__main__":
    success, issues = check_ollama_service()
    if not success:
        print("\n❌ Issues found, please fix according to the above prompts.")
        if issues:
            print("\nIssue Summary:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
    sys.exit(0 if success else 1)

