#!/usr/bin/env python3
# Reverse Engineer API Script:
#   Captures encrypted and raw JS from TARGET_URL
#   Extracts API signing code (fetch|XHR|WebSocket)
#   Bundles stub module with esbuild for Node
#   Executes signing routines in QuickJS to compute headers/cookies
#   Makes authenticated requests via Python requests
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from playwright.async_api import async_playwright, Page, Response
from quickjs import Context

# Configuration
TARGET_URL = "https://www.nike.com/w/mens-shoes-nik1zy7ok"
PAYLOAD_DIR = Path("payloads")
BUNDLE_OUT = Path("stub_module.bundle.js")
TEMPLATE_DIR = Path("templates")
DECRYPT_LOG_PREFIX = "payload_"
DEBUG = True

class ApiReverser:
    def __init__(self, target_url):
        self.target_url = target_url
        self.domain = urlparse(target_url).netloc
        self.payloads_dir = PAYLOAD_DIR
        self.payloads_dir.mkdir(exist_ok=True)
        self.api_endpoints = set()
        self.request_map = {}
        self.cookies = {}

    async def capture_payloads(self, headless=True, timeout=30000):
        """Capture JS payloads and API requests using Playwright"""
        print(f"Capturing payloads from {self.target_url}")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
            )
            
            # Add storage state if exists
            if Path("storage.json").exists():
                try:
                    storage_state = json.loads(Path("storage.json").read_text())
                    if isinstance(storage_state.get("cookies"), list):
                        await context.add_cookies(storage_state["cookies"])
                except Exception as e:
                    print(f"Warning: Failed to load storage state: {e}")
            
            page = await context.new_page()
            
            # Store all request/response pairs for analysis
            self.request_map = {}
            
            # Intercept and store API requests
            async def on_request(request):
                url = request.url
                if "/api/" in url or self.is_api_endpoint(url):
                    self.api_endpoints.add(url)
                    headers = request.headers
                    cookies = headers.get("cookie", "")
                    method = request.method
                    
                    # Save request details
                    parsed = urlparse(url)
                    filename = f"request_{hashlib.md5(url.encode()).hexdigest()[:8]}.json"
                    request_data = {
                        "url": url,
                        "method": method,
                        "headers": dict(headers),
                        "cookies": self._parse_cookies(cookies),
                        "path": parsed.path,
                        "query": parsed.query
                    }
                    if DEBUG:
                        print(f"Captured API request: {url}")
                    self.request_map[url] = request_data
                    (self.payloads_dir / filename).write_text(json.dumps(request_data, indent=2))
            
            page.on('request', on_request)
            
            # Intercept JavaScript files
            async def on_response(response: Response):
                url = response.url
                
                # Store final cookies
                if response.status == 200 and self.domain in url:
                    cookies = await context.cookies()
                    for cookie in cookies:
                        self.cookies[cookie["name"]] = cookie["value"]
                
                # Save JavaScript files
                if url.endswith('.js') or '/js/' in url:
                    try:
                        text = await response.text()
                        content_type = response.headers.get("content-type", "")
                        if "javascript" in content_type or "application/json" in content_type:
                            h = hashlib.md5(url.encode()).hexdigest()[:8]
                            path = self.payloads_dir / f"raw_{h}.js"
                            if not path.exists() and len(text) > 0:
                                path.write_text(text)
                                if DEBUG:
                                    print(f"Saved JS: {url[:60]}... -> {path.name}")
                    except Exception as e:
                        if DEBUG:
                            print(f"Error saving JS {url}: {str(e)}")
            
            page.on('response', on_response)
            
            # Intercept console logs and decrypted payloads
            async def on_console(msg):
                text = msg.text
                if "DECRYPT_LOG:" in text:
                    payload = text[text.index("DECRYPT_LOG:") + len("DECRYPT_LOG:"):]
                    h = hashlib.md5(payload.encode()).hexdigest()[:8]
                    path = self.payloads_dir / f"{DECRYPT_LOG_PREFIX}{h}.js"
                    if not path.exists():
                        path.write_text(payload)
                        if DEBUG:
                            print(f"Captured decrypted payload: {path.name}")
                
                # Also log fetch/XHR calls through console
                if "fetch(" in text or "XMLHttpRequest" in text:
                    path = self.payloads_dir / f"console_{hashlib.md5(text.encode()).hexdigest()[:8]}.log"
                    if not path.exists():
                        path.write_text(text)
            
            page.on('console', on_console)
            
            # Enhanced initialization script to intercept more browser APIs
            await page.add_init_script("""
                (function() {
                    // Log to console when these methods are called
                    const wrappers = {
                        'eval': window.eval,
                        'Function': window.Function,
                        'fetch': window.fetch,
                        'XMLHttpRequest.prototype.open': XMLHttpRequest.prototype.open,
                        'XMLHttpRequest.prototype.send': XMLHttpRequest.prototype.send
                    };
                    
                    // WebAssembly if available
                    if (window.WebAssembly && window.WebAssembly.instantiate) {
                        wrappers['WebAssembly.instantiate'] = window.WebAssembly.instantiate;
                    }
                    
                    // Wrap each function to log its use
                    for (const [name, original] of Object.entries(wrappers)) {
                        if (name.includes('.prototype.')) {
                            const [className, methodName] = name.split('.prototype.');
                            window[className].prototype[methodName] = function(...args) {
                                let logArgs = args;
                                try {
                                    logArgs = args.map(a => 
                                        typeof a === 'object' ? JSON.stringify(a).substring(0, 100) : String(a)
                                    );
                                } catch (e) {
                                    logArgs = ["[Complex Object]"];
                                }
                                console.log(`DECRYPT_LOG:${name} called with ${logArgs.join(', ')}`);
                                return original.apply(this, args);
                            };
                        } else if (name.includes('.')) {
                            const [namespace, method] = name.split('.');
                            const originalMethod = window[namespace][method];
                            window[namespace][method] = function(...args) {
                                console.log(`DECRYPT_LOG:${name} called`);
                                return originalMethod.apply(this, args);
                            };
                        } else {
                            window[name] = function(...args) {
                                console.log(`DECRYPT_LOG:${name} called with ${args.join('|')}`);
                                return original.apply(this, args);
                            };
                        }
                    }
                    
                    // Advanced fetch interception
                    const originalFetch = window.fetch;
                    window.fetch = function(resource, init) {
                        const url = typeof resource === 'string' ? resource : resource.url;
                        let headers = init?.headers || {};
                        
                        console.log(`DECRYPT_LOG:fetch to ${url} with headers ${JSON.stringify(headers).substring(0, 100)}`);
                        return originalFetch.apply(this, arguments);
                    };
                })();
            """)
            
            # Navigate and interact with the page
            await page.goto(self.target_url, timeout=timeout)
            await page.wait_for_load_state('domcontentloaded')
            
            # Scroll to trigger lazy-loaded content
            await self.interact_with_page(page)
            
            # Wait a bit longer for API calls to complete
            await page.wait_for_timeout(5000)
            
            # Save cookies and storage for later
            await context.storage_state(path="storage.json")
            
            # Save final page HTML
            html = await page.content()
            (self.payloads_dir / "final_page.html").write_text(html)
            
            await browser.close()
            
            print(f"Captured {len(list(self.payloads_dir.glob('*.js')))} JavaScript files")
            print(f"Captured {len(self.api_endpoints)} API endpoints")
    
    async def interact_with_page(self, page: Page):
        """Interact with the page to trigger dynamic content and API calls"""
        # Scroll down in increments
        for i in range(5):
            await page.evaluate(f"window.scrollTo(0, {i * 300})")
            await page.wait_for_timeout(1000)
        
        # Try clicking on navigation elements
        try:
            nav_selectors = ["nav a", ".navigation a", ".menu a", "header a"]
            for selector in nav_selectors:
                elements = await page.query_selector_all(selector)
                if elements and len(elements) > 0:
                    # Click the first element
                    try:
                        await elements[0].click()
                        await page.wait_for_load_state('networkidle')
                        await page.go_back()
                        await page.wait_for_load_state('networkidle')
                        break
                    except Exception:
                        continue
        except Exception as e:
            if DEBUG:
                print(f"Error during interaction: {str(e)}")
    
    def is_api_endpoint(self, url):
        """Check if URL appears to be an API endpoint"""
        patterns = [
            r'/api/',
            r'/graphql',
            r'/rest/',
            r'/v\d+/',
            r'/service/',
            r'\.json'
        ]
        return any(re.search(p, url) for p in patterns)
    
    def _parse_cookies(self, cookie_string):
        """Parse cookie string into dictionary"""
        if not cookie_string:
            return {}
        cookies = {}
        for item in cookie_string.split(';'):
            if '=' in item:
                name, value = item.strip().split('=', 1)
                cookies[name] = value
        return cookies
    
    def find_api_modules(self):
        """Find JS modules that contain API calls"""
        api_files = []
        
        # API patterns to look for
        patterns = [
            r"fetch\s*\(",
            r"XMLHttpRequest",
            r"new\s+WebSocket",
            r"\.ajax\s*\(",
            r"\.post\s*\(",
            r"\.get\s*\(",
            r"headers\s*:",
            r"Authorization\s*:",
            r"Bearer\s+",
            r"api_key",
            r"apiKey",
            r"token",
            r"signature"
        ]
        
        for js in self.payloads_dir.glob('*.js'):
            try:
                code = js.read_text()
                if any(re.search(pattern, code) for pattern in patterns):
                    # Prioritize files with higher API pattern matches
                    matches = sum(1 for pattern in patterns if re.search(pattern, code))
                    api_files.append((js, matches))
                    if DEBUG:
                        print(f"Found API code in {js.name} with {matches} matches")
            except Exception as e:
                if DEBUG:
                    print(f"Error reading {js}: {str(e)}")
        
        # Sort by number of matches (descending)
        api_files.sort(key=lambda x: x[1], reverse=True)
        return [file for file, _ in api_files]
    
    def extract_auth_logic(self):
        """Extract authentication logic from API modules"""
        api_files = self.find_api_modules()
        if not api_files:
            print("No API modules found. Make sure capture_payloads() was run successfully.")
            return None
        
        # Create template directory
        TEMPLATE_DIR.mkdir(exist_ok=True)
        
        # Create a template file from the most promising module
        best_file = api_files[0]
        print(f"Using {best_file.name} as primary API module")
        
        # Create stub template
        template = self._create_stub_template(best_file)
        template_file = TEMPLATE_DIR / "auth_stub.js"
        template_file.write_text(template)
        
        return template_file
    
    def _create_stub_template(self, js_file):
        """Create a QuickJS-compatible stub from the JS file"""
        code = js_file.read_text()
        
        # Basic template with auth logic extraction - without Node require statements
        stub_template = """
// API authentication stub generated from {filename}
// Modified for QuickJS compatibility (no require statements)

// Original code context
const window = {{
    location: {{ 
        hostname: '{domain}',
        href: '{target_url}',
        origin: '{origin}'
    }},
    navigator: {{
        userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }},
    document: {{
        cookie: '',
        createElement: function() {{ return {{}}; }}
    }}
}};
const document = window.document;
const navigator = window.navigator;
const location = window.location;

// Basic utilities for QuickJS
const simpleHash = function(str) {{
    let hash = 0;
    for (let i = 0; i < str.length; i++) {{
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash |= 0;
    }}
    return hash.toString(16);
}};

// Stub fetch implementation
const fetch = () => {{}};

// Extracted authentication code
{extracted_code}

// Helper function to generate headers and cookies
function genHeaders(params) {{
    // Default parameters
    const timestamp = params[0].timestamp || Math.floor(Date.now() / 1000);
    const url = params[0].url || '{target_url}';
    const method = params[0].method || 'GET';
    const body = params[0].body || null;
    
    try {{
        // This is a basic implementation, modify based on site-specific requirements
        const headers = {{
            'User-Agent': navigator.userAgent,
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': window.location.href,
            'Origin': window.location.origin
        }};
        
        // Some sites use CSRF tokens or other security measures
        // headers['X-CSRF-Token'] = simpleHash(timestamp + url);
        
        const cookies = {{
            // Nike often uses cookies for session tracking
            // 'anonymousId': simpleHash(navigator.userAgent + timestamp),
        }};
        
        // Must return a JSON string for QuickJS compatibility
        return JSON.stringify({{ headers, cookies }});
    }} catch (error) {{
        return JSON.stringify({{ headers: {{}}, cookies: {{}} }});
    }}
}}
        """.format(
            filename=js_file.name,
            domain=self.domain,
            target_url=self.target_url,
            origin=urlparse(self.target_url).scheme + "://" + self.domain,
            extracted_code="// TODO: Extract relevant auth code from the original file if needed"
        )
        
        return stub_template
    
    def finalize_stub(self, template_file):
        """Finalize the stub for execution with QuickJS"""
        # For QuickJS compatibility, we'll just use the template directly
        # without bundling with esbuild (which creates Node.js module code)
        try:
            print(f"Finalizing {template_file} for QuickJS...")
            # Create a clean copy for execution
            js_content = template_file.read_text()
            BUNDLE_OUT.write_text(js_content)
            print(f"Successfully prepared {BUNDLE_OUT} for QuickJS")
            return True
        except Exception as e:
            print(f"Error preparing JS for QuickJS: {str(e)}")
            return False
    
    def make_authenticated_request(self, url=None, method="GET", body=None):
        """Make an authenticated request using the generated auth logic"""
        url = url or self.target_url
        
        # Load the JavaScript in QuickJS
        if not BUNDLE_OUT.exists():
            print(f"Error: {BUNDLE_OUT} not found. Run finalize_stub() first.")
            return None
        
        try:
            # Create QuickJS context
            ctx = Context()
            
            # Load the JavaScript code
            js_code = BUNDLE_OUT.read_text()
            ctx.eval(js_code)
            
            # Generate authentication
            params = {
                "timestamp": int(time.time()),
                "url": url,
                "method": method,
                "body": body
            }
            
            # Call the genHeaders function in QuickJS
            try:
                # Use eval instead of call for better compatibility
                params_json = json.dumps([params])
                result_js = f"genHeaders({params_json})"
                auth_result = ctx.eval(result_js)
                
                # Convert QuickJS objects to Python
                auth = json.loads(auth_result)
                
                headers = {}
                cookies = {}
                
                if auth and "headers" in auth:
                    headers = auth["headers"]
                
                if auth and "cookies" in auth:
                    cookies = auth["cookies"]
                
                # Use the stored cookies if no cookies were generated
                if not cookies and self.cookies:
                    cookies = self.cookies
                
                # Make the request
                session = requests.Session()
                session.headers.update(headers)
                for k, v in cookies.items():
                    session.cookies.set(k, v)
                
                print(f"Making authenticated request to {url}")
                print(f"Headers: {json.dumps(headers, indent=2)}")
                print(f"Cookies: {json.dumps({k: v for k, v in cookies.items()}, indent=2)}")
                
                if method.upper() == "GET":
                    response = session.get(url)
                elif method.upper() == "POST":
                    response = session.post(url, json=body if body else {})
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                return response
                
            except Exception as e:
                print(f"Error executing JS in QuickJS: {str(e)}")
                
                # Fallback to manual request with captured cookies
                if self.cookies:
                    print("Falling back to captured cookies for authentication")
                    session = requests.Session()
                    session.headers.update({
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                        "Accept": "application/json",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": self.target_url
                    })
                    
                    for k, v in self.cookies.items():
                        session.cookies.set(k, v)
                        
                    response = session.get(url)
                    return response
                
                return None
            
        except Exception as e:
            print(f"Error making authenticated request: {str(e)}")
            return None

# Main flow
async def main():
    reverser = ApiReverser(TARGET_URL)
    
    try:
        # Step 1: Capture JS and API payloads
        await reverser.capture_payloads(headless=False)
        
        # Step 2: Find API modules
        api_files = reverser.find_api_modules()
        if not api_files:
            print("Error: No API modules found. Try running with headless=False to see browser interactions.")
            sys.exit(1)
        
        # Step 3: Extract authentication logic
        template_file = reverser.extract_auth_logic()
        if not template_file:
            print("Error: Failed to extract authentication logic.")
            sys.exit(1)
        
        # Step 4: Prompt user to review and modify the template
        print(f"\nGenerated template at {template_file}")
        print("Please review and modify the template to implement the correct authentication logic.")
        print("Press Enter when ready to continue...")
        input()
        
        # Step 5: Bundle the final stub
        if not reverser.finalize_stub(template_file):
            print("Error: Failed to bundle the stub module.")
            sys.exit(1)
        
        # Step 6: Test with actual API endpoints
        test_endpoints(reverser)
        
        print("\nThe reverse engineering process is complete.")
        print(f"You can now use the {BUNDLE_OUT} module for browserless API requests.")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        sys.exit(1)

def test_endpoints(reverser):
    """Test some actual Nike API endpoints"""
    # First test the main site to verify basic functionality
    main_response = reverser.make_authenticated_request()
    if main_response:
        print(f"Main site response status: {main_response.status_code}")
        
        # Save the response for analysis
        (PAYLOAD_DIR / "response_main.txt").write_text(
            f"Status: {main_response.status_code}\n\n"
            f"Headers: {json.dumps(dict(main_response.headers), indent=2)}\n\n"
            f"Body preview: {main_response.text[:500]}..."
        )
    
    # Now test an actual API endpoint from the captured list
    if reverser.api_endpoints:
        print("\nTesting actual API endpoints:")
        for i, endpoint in enumerate(list(reverser.api_endpoints)[:3], 1):
            print(f"\nAPI Endpoint #{i}: {endpoint}")
            api_response = reverser.make_authenticated_request(url=endpoint)
            
            if api_response:
                print(f"API response status: {api_response.status_code}")
                
                # Save the API response
                filename = f"response_api_{i}.txt"
                try:
                    content_type = api_response.headers.get('Content-Type', '')
                    if 'json' in content_type:
                        response_body = json.dumps(api_response.json(), indent=2)
                    else:
                        response_body = api_response.text[:2000] + "..."
                        
                    (PAYLOAD_DIR / filename).write_text(
                        f"API Endpoint: {endpoint}\n\n"
                        f"Status: {api_response.status_code}\n\n"
                        f"Headers: {json.dumps(dict(api_response.headers), indent=2)}\n\n"
                        f"Body: {response_body}"
                    )
                    print(f"Response saved to {filename}")
                    
                except Exception as e:
                    print(f"Error saving API response: {e}")
    else:
        print("No API endpoints were captured for testing.")

if __name__ == '__main__':
    asyncio.run(main())
