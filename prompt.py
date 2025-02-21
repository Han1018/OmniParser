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

CN_FIND_BBOX_FROM_LIST_IMAGE_PROMPT = """你是一個 AI 助手，負責根據以下輸入來找出與文字輸入最相關的項目及其對應的 bbox:
1. 文字輸入 Input: 目標字串，用以查找對應項目
2. 標註後的圖片 Labeled Image: 可協助理解圖片中各項目的語意與位置。
3. 項目列表 List: 一個 JSON Array，每個元素代表圖片中可點擊的內容，其格式如下：
```json
{
"type": "icon" | "text",
"bbox": [x1, y1, x2, y2],
"content": "內容"
}
```

### 處理方式：
1. 找到其中最相關的一個內容
    - 請依據標註後的圖片 (Labeled Image) 中呈現的語意，理解圖片內容後找出與 Input 語意最相關的項目，回傳 List 中的 bbox。

2. 只需要回傳一個結果。
    - 若有多個候選，以「最相關」(或相似度最高) 為準。

3. 回答只需要回答 JSON 內容，不需要包含解釋、思考過程或多餘的文字。

4. 回傳結果：
   - 若找到匹配的內容，回傳：
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

### 範例輸入：
- Input: "settings"
- Labeled Image: (這裡省略真實影像或 Base64 等資料，只需知道該圖經過標註，每個項目與 BBox 都對應到這張圖)
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

### Expected Output:
- 若 Input 與某個項目的 content 完全一致（例如 "settings"），則回傳：
```json
{
  "bbox": [0.015625, 0.21296297013759613, 0.046875, 0.24259258806705475],
  "content": "settings"
}
```
- 若無完全匹配的項目，但可根據圖片語意判斷出與 Input 最相關的內容，則回傳該項目的 bbox 與內容。
- 若無法判斷任何明顯相關的內容，則回傳：
```json
null
```
"""

CN_FIND_BBOX_FROM_LIST_IMAGE_PROMPT_UPDATE = """你是一個 AI 助手，負責根據以下輸入，來找出圖片中與文字輸入最相關的項目及其對應的 bbox:
1. 文字輸入 Input: 目標字串，用以查找對應項目
2. 標註後的圖片 Labeled Image: 標記圖片 text / 可點擊 icon 的位置(皆有 id 標示對應到 List 中的 item)
3. 項目列表 List: 一個 JSON Array，每個元素代表圖片中可點擊的內容，其格式如下：
```json
{
"type": "icon" | "text",
"bbox": [x1, y1, x2, y2],
"content": "內容"
}
```

### 處理方式：
0. 查看圖片中哪一個 ID 最符合目標，然後我回傳對應的 bbox

1. 找到其中最相關的一個內容
    - 請依據標註後的圖片 (Labeled Image)，找出與 Input 語意最相關的項目，回傳 List 中相同 id 的 bbox。
    - Input 描述的內容一定會在圖片中出現，不會有其他情況。

2. 只需要回傳一個結果。
    - 若有多個候選，以「最相關」(或相似度最高) 為準。

3. 回答只需要回答 JSON 內容，不需要包含解釋、思考過程或多餘的文字。

4. 回傳結果：
   - 若找到匹配的內容，回傳：
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

### 範例輸入：
- Input: "settings"
- Labeled Image: (這裡省略真實影像或 Base64 等資料，只需知道該圖經過標註，每個項目與 BBox 都對應到這張圖)
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

### Expected Output:
- 若 Input 與 List 某個項目的 content 完全一致（例如 "settings"），則回傳：
```json
{
  "bbox": [0.015625, 0.21296297013759613, 0.046875, 0.24259258806705475],
  "content": "settings"
}
```
- 若無完全匹配的項目，根據圖片查看那一個 id 與 Input 最相關，回傳 List 對應 id 的 bbox 與內容。
- 若無法判斷任何明顯相關的內容，則回傳：
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