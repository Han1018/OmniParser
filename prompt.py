FIND_BBOX_FROM_LIST_PROMPT = """You are an AI assistant tasked with extracting relevant content from a list of items based on a given input. Each item in the list follows the format:

```json
{
  "type": "icon" | "text",
  "bbox": [x1, y1, x2, y2],
  "interactivity": true | false,
  "content": "View",
  "source": "box_ocr_content_ocr" | "box_yolo_content_yolo"
}
```

### Instructions:
1. **Input:** You will receive a text input string and list of items.
2. **Processing:** Search through the list of items to find the most relevant content that matches or relates to the input.
3. **Criteria:**
   - Prioritize exact matches first.
   - If no exact match is found, look for partial matches or semantically similar content.
   - Consider the relevance of "text" type items over "icon" type items if both are available.
4. **Output:**
   - If a relevant match is found, return a JSON object containing:
     ```json
     {
       "bbox": [x1, y1, x2, y2],
       "content": "Matched content"
     }
     ```
   - If no relevant match is found, return:
     ```json
     null
     ```

### Example:
- Input: "Settings"

- Items List:
```json
[
  {"type": "text", "bbox": [10, 20, 50, 60], "interactivity": true, "content": "Settings", "source": "box_ocr_content_ocr"},
  {"type": "icon", "bbox": [15, 25, 55, 65], "interactivity": true, "content": "Gear", "source": "box_yolo_content_yolo"},
  {"type": "text", "bbox": [30, 40, 70, 80], "interactivity": false, "content": "Preferences", "source": "box_ocr_content_ocr"}
]
```

- Expected Output:
```json
{
  "bbox": [10, 20, 50, 60],
  "content": "Settings"
}
```

If no matching content is found, return:
```json
null
```
"""
CN_FIND_BBOX_FROM_LIST_PROMPT = """你是一個 AI 助理，負責從 List item 中找出與輸入文字最相關的 item。

### **輸入格式：**
- 你會收到一個文字輸入 Input 和一個 List，List 中每個 item 包含：
  ```json
  {
    "type": "icon" | "text",
    "bbox": [x1, y1, x2, y2],
    "content": "..."
  }
  ```

### **處理方式：**
1. 這是一個簡單的任務，負責比較 item['content'] 與輸入文字:
   - 如果完全相同，選擇那個 item。
   - 如果沒有完全相同的，找出最接近的

2. 只需要回傳一個 item 結果。

3. 你只需要比較 item['content'] 與輸入文字, 不須使用任何其他工具

4. 回答只需要回答 json 內容，不需要包含其他任何細節/思考過程。

5. **輸出格式：**
   - 若找到匹配的內容，只須回傳 item bbox 和 content：
     ```json
     {
       "bbox": [x1, y1, x2, y2],
       "content": "找到的內容"
     }
     ```
   - 若找不到，回傳：
     ```json
     null
     ```

### **範例：**
- List:
```json
[
  {
    "type": "text",
    "bbox": [0.015625, 0.21296297013759613, 0.046875, 0.24259258806705475],
    "interactivity": "false",
    "content": "settings",
    "source": "box_ocr_content_ocr"
  },
  {
    "type": "text",
    "bbox": [0.0, 0.2648148238658905, 0.03750000149011612, 0.3037036955356598],
    "interactivity": "false",
    "content": "website",
    "source": "box_ocr_content_ocr"
  },
  {
    "type": "text",
    "bbox": [0.6635416746139526, 0.27037036418914795, 0.7697916626930237, 0.3018518388271332],
    "interactivity": "false",
    "content": " Search New folder",
    "source": "box_ocr_content_ocr"
  },
]
```

#### Expected Output:
```json
{
  "bbox": [0.015625, 0.21296297013759613, 0.046875, 0.24259258806705475],
  "content": "settings"
}
```

若找不到匹配內容，回傳：
```json
null
```
"""

GET_OUTPUT_PROMPT = """You are a JSON extractor. Your task is to analyze the input, which may contain multiple JSON objects and the final answer. Extract the final JSON object and return it as the sole output.

Rules:
1. Identify the final JSON object in the input.
2. Return only the extracted JSON object.
3. If no valid JSON object is found, return `null`.

Expected Output Format:
- If a valid JSON object is found:
  ```json
  {
    "bbox": [0.015625, 0.21296297013759613, 0.046875, 0.24259258806705475],
    "content": "settings"
  }
  ```
- If no valid JSON object is found:
  ```json
  null
  ```
"""