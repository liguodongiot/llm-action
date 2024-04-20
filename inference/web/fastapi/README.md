
```
from typing import Optional
from pydantic import BaseModel, Field
 
# 定义请求的数据模型
class Item(BaseModel):
   name: str = Field(..., description="物品名称")
   description: Optional[str] = Field(None, description="物品描述")
   price: float = Field(..., description="物品价格")


from fastapi import FastAPI, HTTPException, Request
 
app = FastAPI()
 
# 创建POST路由
@app.post("/items/")
async def create_item(item: Item):
   # 这里你可以处理接收到的数据，比如保存到数据库
   # 作为示例，我们只是打印出来
   print(item)
   


uvicorn main:app --reload
```