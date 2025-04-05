# MultiAgent-Meeting MVP (使用Deepseek)

这是基于CrewAI框架实现的多智能体会议系统的最小可行产品(MVP)版本，使用Deepseek模型作为底层LLM。该系统模拟了不同角色的专业人士在项目立项会议中的协作过程。

## 设置环境

1. 安装依赖：

```bash
# 使用清华镜像源安装依赖
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. 配置API密钥：

   - 复制`.env.example`文件为`.env`
   - 在`.env`文件中填入你的Deepseek API密钥

```bash
cp .env.example .env
# 然后编辑.env文件，填入你的Deepseek API密钥
```

## 运行MVP

执行以下命令运行模拟会议：

```bash
python mvp.py
```

系统将模拟一次完整的项目立项会议过程，包含以下角色：
- 产品经理：负责定义产品方案
- 技术架构师：负责设计技术实现方案
- 市场分析师：负责提供市场洞察
- 用户体验设计师：负责设计用户体验

会议将按顺序完成以下任务：
1. 市场分析
2. 产品定义
3. 技术方案设计
4. 用户体验设计
5. 项目方案汇总

## 输出结果

会议结束后，系统会：
1. 在控制台输出最终的项目方案
2. 将完整方案保存到`meeting_result.md`文件中

## 自定义会议主题

如需更改会议主题，请编辑`mvp.py`文件中的`MEETING_TOPIC`变量：

```python
# 定义会议主题
MEETING_TOPIC = "你的自定义会议主题"
```

## Deepseek模型配置

如果需要调整Deepseek模型的参数，可以修改`mvp.py`文件中的`deepseek_llm`配置：

```python
deepseek_llm = LLM(
    model="custom_openai/deepseek-chat",  # 模型名称
    base_url=os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1"),  # API基础URL
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),  # API密钥
    temperature=0.7,  # 温度参数
    max_tokens=4000,  # 最大生成token数
)
```

## 扩展方向

这个MVP可以按以下方向扩展：
- 添加更多专业角色
- 实现更复杂的会议流程
- 添加知识库集成
- 实现可视化会议记录
- 添加冲突解决机制
- 优化输出格式 