#!/usr/bin/env python3
"""
AWS Deployment Testing Script for Evidence Indicator RAG System
Tests both Lambda and ECS/Fargate deployments
"""

import requests
import json
import time
import sys
import os
from typing import Dict, List, Any
from datetime import datetime

class AWSDeploymentTester:
    def __init__(self, api_url: str = None):
        self.api_url = api_url
        self.test_results = []
        self.start_time = datetime.now()
        
    def log_test(self, test_name: str, success: bool, details: str = "", duration: float = 0):
        """Log test results"""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({duration:.2f}s)")
        if details:
            print(f"   📝 {details}")
    
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        start_time = time.time()
        try:
            if self.api_url:
                response = requests.get(f"{self.api_url}/health", timeout=10)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test("Health Check", True, f"Service: {data.get('service', 'Unknown')}", duration)
                    return True
                else:
                    self.log_test("Health Check", False, f"Status: {response.status_code}", duration)
                    return False
            else:
                self.log_test("Health Check", False, "No API URL provided", 0)
                return False
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Health Check", False, f"Error: {str(e)}", duration)
            return False
    
    def test_single_query(self, query: str, expected_keywords: List[str] = None) -> bool:
        """Test single query endpoint"""
        start_time = time.time()
        try:
            if not self.api_url:
                self.log_test(f"Query: {query[:30]}...", False, "No API URL provided", 0)
                return False
            
            payload = {"query": query}
            response = requests.post(
                f"{self.api_url}/query",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', '')
                processing_time = data.get('processing_time', 0)
                
                # Check if answer contains expected keywords
                success = True
                details = f"Processing time: {processing_time:.2f}s"
                
                if expected_keywords:
                    missing_keywords = [kw for kw in expected_keywords if kw not in answer]
                    if missing_keywords:
                        success = False
                        details += f", Missing keywords: {missing_keywords}"
                
                self.log_test(f"Query: {query[:30]}...", success, details, duration)
                return success
            else:
                self.log_test(f"Query: {query[:30]}...", False, f"Status: {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(f"Query: {query[:30]}...", False, f"Error: {str(e)}", duration)
            return False
    
    def test_batch_query(self, queries: List[str]) -> bool:
        """Test batch query endpoint"""
        start_time = time.time()
        try:
            if not self.api_url:
                self.log_test("Batch Query", False, "No API URL provided", 0)
                return False
            
            payload = {"queries": queries}
            response = requests.post(
                f"{self.api_url}/batch_query",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                total_queries = data.get('total_queries', 0)
                avg_processing_time = data.get('average_processing_time', 0)
                
                details = f"Total queries: {total_queries}, Avg time: {avg_processing_time:.2f}s"
                self.log_test("Batch Query", True, details, duration)
                return True
            else:
                self.log_test("Batch Query", False, f"Status: {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Batch Query", False, f"Error: {str(e)}", duration)
            return False
    
    def test_performance(self, query: str, expected_max_time: float = 5.0) -> bool:
        """Test performance with multiple requests"""
        start_time = time.time()
        try:
            if not self.api_url:
                self.log_test("Performance Test", False, "No API URL provided", 0)
                return False
            
            times = []
            success_count = 0
            total_requests = 5
            
            for i in range(total_requests):
                request_start = time.time()
                try:
                    response = requests.post(
                        f"{self.api_url}/query",
                        json={"query": query},
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )
                    request_time = time.time() - request_start
                    times.append(request_time)
                    
                    if response.status_code == 200:
                        success_count += 1
                    
                    # Small delay between requests
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"   ⚠️  Request {i+1} failed: {str(e)}")
            
            duration = time.time() - start_time
            avg_time = sum(times) / len(times) if times else 0
            max_time = max(times) if times else 0
            
            success = success_count == total_requests and max_time <= expected_max_time
            details = f"Success: {success_count}/{total_requests}, Avg: {avg_time:.2f}s, Max: {max_time:.2f}s"
            
            self.log_test("Performance Test", success, details, duration)
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Performance Test", False, f"Error: {str(e)}", duration)
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid requests"""
        start_time = time.time()
        try:
            if not self.api_url:
                self.log_test("Error Handling", False, "No API URL provided", 0)
                return False
            
            # Test empty query
            response1 = requests.post(
                f"{self.api_url}/query",
                json={"query": ""},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            # Test missing query
            response2 = requests.post(
                f"{self.api_url}/query",
                json={},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            # Test invalid JSON
            response3 = requests.post(
                f"{self.api_url}/query",
                data="invalid json",
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            duration = time.time() - start_time
            
            # All should return 400 status
            success = (response1.status_code == 400 and 
                      response2.status_code == 400 and 
                      response3.status_code == 400)
            
            details = f"Empty: {response1.status_code}, Missing: {response2.status_code}, Invalid: {response3.status_code}"
            self.log_test("Error Handling", success, details, duration)
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Error Handling", False, f"Error: {str(e)}", duration)
            return False
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("🧪 Evidence Indicator RAG - AWS Deployment Testing")
        print("=" * 60)
        print(f"🌐 API URL: {self.api_url or 'Not provided'}")
        print(f"⏰ Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test queries
        test_queries = [
            ("コンバインとは何ですか", ["コンバイン", "農業機械"]),
            ("音位転倒について説明してください", ["音位転倒"]),
            ("What is a combine harvester?", ["combine", "harvester"]),
            ("AI技術の最新動向", ["AI", "技術"]),
            ("機械学習の応用例", ["機械学習", "応用"])
        ]
        
        # Run tests
        tests = [
            ("Health Check", self.test_health_check),
            ("Error Handling", self.test_error_handling),
            ("Performance Test", lambda: self.test_performance("コンバインとは何ですか", 5.0)),
            ("Batch Query", lambda: self.test_batch_query([q[0] for q in test_queries[:3]]))
        ]
        
        # Add individual query tests
        for query, expected_keywords in test_queries:
            tests.append((f"Query: {query[:30]}...", lambda q=query, kw=expected_keywords: self.test_single_query(q, kw)))
        
        # Execute tests
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                self.log_test(test_name, False, f"Exception: {str(e)}", 0)
        
        # Generate summary
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        summary = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": (passed / total) * 100 if total > 0 else 0,
            "total_duration": total_duration,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "test_results": self.test_results
        }
        
        # Print summary
        print()
        print("📊 Test Summary")
        print("=" * 60)
        print(f"✅ Passed: {passed}/{total}")
        print(f"❌ Failed: {total - passed}/{total}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️  Total Duration: {total_duration:.2f}s")
        print(f"🕐 Start Time: {self.start_time.strftime('%H:%M:%S')}")
        print(f"🕐 End Time: {end_time.strftime('%H:%M:%S')}")
        
        if passed == total:
            print("🎉 All tests passed! Deployment is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the deployment.")
        
        return summary

def main():
    """Main function"""
    # Get API URL from command line or environment
    api_url = None
    
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    else:
        api_url = os.environ.get('EVIDENCE_INDICATOR_API_URL')
    
    if not api_url:
        print("❌ No API URL provided!")
        print("Usage: python test_aws_deployment.py <API_URL>")
        print("Or set EVIDENCE_INDICATOR_API_URL environment variable")
        print()
        print("Examples:")
        print("  python test_aws_deployment.py https://abc123.execute-api.us-east-1.amazonaws.com/prod")
        print("  python test_aws_deployment.py http://your-alb-dns-name")
        sys.exit(1)
    
    # Run tests
    tester = AWSDeploymentTester(api_url)
    summary = tester.run_comprehensive_test()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"aws_deployment_test_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Test results saved to: {results_file}")
    
    # Exit with appropriate code
    if summary['passed'] == summary['total_tests']:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 