# Playwright MCP Tools

## Interactions

### browser_snapshot
- **Title:** Page snapshot  
- **Description:** Capture accessibility snapshot of the current page, this is better than screenshot  
- **Parameters:** None  
- **Read-only:** true

---

### browser_click
- **Title:** Click  
- **Description:** Perform click on a web page  
- **Parameters:**  
  - `element` (string): Human-readable element description used to obtain permission to interact with the element  
  - `ref` (string): Exact target element reference from the page snapshot  
- **Read-only:** false

---

### browser_drag
- **Title:** Drag mouse  
- **Description:** Perform drag and drop between two elements  
- **Parameters:**  
  - `startElement` (string): Human-readable source element description  
  - `startRef` (string): Exact source element reference  
  - `endElement` (string): Human-readable target element description  
  - `endRef` (string): Exact target element reference  
- **Read-only:** false

---

### browser_hover
- **Title:** Hover mouse  
- **Description:** Hover over element on page  
- **Parameters:**  
  - `element` (string): Human-readable element description  
  - `ref` (string): Exact target element reference  
- **Read-only:** true

---

### browser_type
- **Title:** Type text  
- **Description:** Type text into editable element  
- **Parameters:**  
  - `element` (string): Human-readable element description  
  - `ref` (string): Exact target element reference  
  - `text` (string): Text to type  
  - `submit` (boolean, optional): Whether to submit entered text (press Enter after)  
  - `slowly` (boolean, optional): Type one character at a time  
- **Read-only:** false

---

### browser_select_option
- **Title:** Select option  
- **Description:** Select an option in a dropdown  
- **Parameters:**  
  - `element` (string): Human-readable element description  
  - `ref` (string): Exact target element reference  
  - `values` (array): Values to select  
- **Read-only:** false

---

### browser_press_key
- **Title:** Press a key  
- **Description:** Press a key on the keyboard  
- **Parameters:**  
  - `key` (string): Name of the key or character  
- **Read-only:** false

---

### browser_wait_for
- **Title:** Wait for  
- **Description:** Wait for text to appear/disappear or a specified time  
- **Parameters:**  
  - `time` (number, optional): Time to wait (seconds)  
  - `text` (string, optional): Text to wait for  
  - `textGone` (string, optional): Text to wait to disappear  
- **Read-only:** true

---

### browser_file_upload
- **Title:** Upload files  
- **Description:** Upload one or multiple files  
- **Parameters:**  
  - `paths` (array): Absolute paths to files  
- **Read-only:** false

---

### browser_handle_dialog
- **Title:** Handle a dialog  
- **Description:** Handle a dialog  
- **Parameters:**  
  - `accept` (boolean): Whether to accept  
  - `promptText` (string, optional): Text for prompt dialog  
- **Read-only:** false

---

## Navigation

### browser_navigate
- **Title:** Navigate to a URL  
- **Description:** Navigate to a URL  
- **Parameters:**  
  - `url` (string): The URL  
- **Read-only:** false

---

### browser_navigate_back
- **Title:** Go back  
- **Description:** Go back to the previous page  
- **Parameters:** None  
- **Read-only:** true

---

### browser_navigate_forward
- **Title:** Go forward  
- **Description:** Go forward to the next page  
- **Parameters:** None  
- **Read-only:** true

---

## Resources

### browser_take_screenshot
- **Title:** Take a screenshot  
- **Description:** Take a screenshot of the current page  
- **Parameters:**  
  - `raw` (boolean, optional): Return PNG (true) or JPEG (false, default)  
  - `filename` (string, optional): File name  
  - `element` (string, optional): Element description  
  - `ref` (string, optional): Element reference  
- **Read-only:** true

---

### browser_pdf_save
- **Title:** Save as PDF  
- **Description:** Save page as PDF  
- **Parameters:**  
  - `filename` (string, optional): File name  
- **Read-only:** true

---

### browser_network_requests
- **Title:** List network requests  
- **Description:** Returns all network requests since loading the page  
- **Parameters:** None  
- **Read-only:** true

---

### browser_console_messages
- **Title:** Get console messages  
- **Description:** Returns all console messages  
- **Parameters:** None  
- **Read-only:** true

---

## Utilities

### browser_install
- **Title:** Install the browser  
- **Description:** Install the browser specified in the config  
- **Parameters:** None  
- **Read-only:** false

---

### browser_close
- **Title:** Close browser  
- **Description:** Close the page  
- **Parameters:** None  
- **Read-only:** true

---

### browser_resize
- **Title:** Resize browser window  
- **Description:** Resize the browser window  
- **Parameters:**  
  - `width` (number): Width  
  - `height` (number): Height  
- **Read-only:** true

---

## Tabs

### browser_tab_list
- **Title:** List tabs  
- **Description:** List browser tabs  
- **Parameters:** None  
- **Read-only:** true

---

### browser_tab_new
- **Title:** Open a new tab  
- **Description:** Open a new tab  
- **Parameters:**  
  - `url` (string, optional): URL for new tab  
- **Read-only:** true

---

### browser_tab_select
- **Title:** Select a tab  
- **Description:** Select a tab by index  
- **Parameters:**  
  - `index` (number): Tab index  
- **Read-only:** true

---

### browser_tab_close
- **Title:** Close a tab  
- **Description:** Close a tab  
- **Parameters:**  
  - `index` (number, optional): Tab index (current if not provided)  
- **Read-only:** false

---

## Testing

### browser_generate_playwright_test
- **Title:** Generate a Playwright test  
- **Description:** Generate a Playwright test for given scenario  
- **Parameters:**  
  - `name` (string): Test name  
  - `description` (string): Test description  
  - `steps` (array): Test steps  
- **Read-only:** true

---

## Vision mode

### browser_screen_capture
- **Title:** Take a screenshot  
- **Description:** Take a screenshot of the current page  
- **Parameters:** None  
- **Read-only:** true

---

### browser_screen_move_mouse
- **Title:** Move mouse  
- **Description:** Move mouse to a given position  
- **Parameters:**  
  - `element` (string): Element description  
  - `x` (number): X coordinate  
  - `y` (number): Y coordinate  
- **Read-only:** true

---

### browser_screen_click
- **Title:** Click  
- **Description:** Click left mouse button  
- **Parameters:**  
  - `element` (string): Element description  
  - `x` (number): X coordinate  
  - `y` (number): Y coordinate  
- **Read-only:** false

---

### browser_screen_drag
- **Title:** Drag mouse  
- **Description:** Drag left mouse button  
- **Parameters:**  
  - `element` (string): Element description  
  - `startX` (number): Start X  
  - `startY` (number): Start Y  
  - `endX` (number): End X  
  - `endY` (number): End Y  
- **Read-only:** false

---

### browser_screen_type
- **Title:** Type text  
- **Description:** Type text  
- **Parameters:**  
  - `text` (string): Text to type  
  - `submit` (boolean, optional): Submit after typing  
- **Read-only:** false

---

### browser_press_key
- **Title:** Press a key  
- **Description:** Press a key on the keyboard  
- **Parameters:**  
  - `key` (string): Name of the key or character  
- **Read-only:** false

---

### browser_wait_for
- **Title:** Wait for  
- **Description:** Wait for text to appear/disappear or a specified time  
- **Parameters:**  
  - `time` (number, optional): Time to wait (seconds)  
  - `text` (string, optional): Text to wait for  
  - `textGone` (string, optional): Text to wait to disappear  
- **Read-only:** true

---

### browser_file_upload
- **Title:** Upload files  
- **Description:** Upload one or multiple files  
- **Parameters:**  
  - `paths` (array): Absolute paths to files  
- **Read-only:** false

---

### browser_handle_dialog
- **Title:** Handle a dialog  
- **Description:** Handle a dialog  
- **Parameters:**  
  - `accept` (boolean): Whether to accept  
  - `promptText` (string, optional): Text for prompt dialog  
- **Read-only:** false