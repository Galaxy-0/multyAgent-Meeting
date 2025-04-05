import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from textwrap import dedent
from crewai import LLM

# 加载环境变量
load_dotenv()

# 配置Deepseek模型
deepseek_llm = LLM(
    model="deepseek-reasoner",  # 修改模型名称
    base_url=os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1"),
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    temperature=1.0,
    max_tokens=8000,
)

# 定义会议主题
MEETING_TOPIC = "开发一个基于AI的个人读书助理应用"

# 定义智能体（会议参与者）
def create_agents():
    # 产品经理
    product_manager = Agent(
        role="产品经理",
        goal="定义一个创新且可行的产品方案",
        backstory=dedent("""
        你是一位经验丰富的产品经理，擅长捕捉市场需求并将其转化为产品功能。
        你需要在会议中主导产品定义环节，确保产品既有创新性又能满足用户实际需求。
        """),
        verbose=True,
        llm=deepseek_llm
    )

    # 技术架构师
    tech_architect = Agent(
        role="技术架构师",
        goal="设计技术可行且高效的系统架构",
        backstory=dedent("""
        你是一位资深技术架构师，精通各种技术栈和架构模式。
        你需要评估产品需求的技术可行性，并提出最佳的技术实现方案。
        """),
        verbose=True,
        llm=deepseek_llm
    )

    # 市场分析师
    market_analyst = Agent(
        role="市场分析师",
        goal="提供准确的市场洞察和竞品分析",
        backstory=dedent("""
        你是一位敏锐的市场分析师，擅长研究市场趋势和竞争对手情况。
        你需要分析目标市场的需求和机会，以及类似产品的优缺点。
        """),
        verbose=True,
        llm=deepseek_llm
    )

    # 用户体验设计师
    ux_designer = Agent(
        role="用户体验设计师",
        goal="确保产品具有出色的用户体验",
        backstory=dedent("""
        你是一位富有创意的用户体验设计师，擅长设计直观且吸引人的交互体验。
        你需要考虑产品的易用性和用户满意度，提出设计建议。
        """),
        verbose=True,
        llm=deepseek_llm
    )

    return [product_manager, tech_architect, market_analyst, ux_designer]

# 定义任务
def create_tasks(agents):
    product_manager, tech_architect, market_analyst, ux_designer = agents
    
    # 市场分析任务
    market_analysis = Task(
        description=f"""
        分析'{MEETING_TOPIC}'的市场机会:
        1. 目标市场规模和增长趋势
        2. 主要竞争对手及其优缺点
        3. 潜在用户痛点和需求
        4. 市场差异化机会

        输出一份简短的市场分析报告。
        """,
        agent=market_analyst,
        expected_output="市场分析报告，包含市场规模、竞争格局、用户需求和差异化机会"
    )
    
    # 产品定义任务
    product_definition = Task(
        description=f"""
        基于市场分析，定义'{MEETING_TOPIC}'的产品方案:
        1. 核心功能和特性
        2. 目标用户群体
        3. 价值主张
        4. 产品愿景和目标

        参考市场分析的结果来完成这项任务。
        输出一份产品定义文档。
        """,
        agent=product_manager,
        expected_output="产品定义文档，包含功能列表、目标用户、价值主张和产品愿景",
        context=[market_analysis]
    )
    
    # 技术方案任务
    tech_solution = Task(
        description=f"""
        基于产品定义，设计'{MEETING_TOPIC}'的技术方案:
        1. 系统架构
        2. 技术栈选择
        3. 核心功能实现方法
        4. 技术风险和解决方案

        参考产品定义的结果来完成这项任务。
        输出一份技术实现方案。
        """,
        agent=tech_architect,
        expected_output="技术实现方案，包含系统架构、技术栈、实现方法和风险对策",
        context=[product_definition]
    )
    
    # 用户体验设计任务
    ux_design = Task(
        description=f"""
        基于产品定义和技术方案，提出'{MEETING_TOPIC}'的用户体验设计:
        1. 用户界面关键元素
        2. 用户流程和交互
        3. 情感化设计考虑
        4. 易用性原则

        参考产品定义和技术方案的结果来完成这项任务。
        输出一份用户体验设计概要。
        """,
        agent=ux_designer,
        expected_output="用户体验设计概要，包含界面元素、用户流程、情感化设计和易用性原则",
        context=[product_definition, tech_solution]
    )
    
    # 项目方案汇总任务
    project_proposal = Task(
        description=f"""
        整合所有分析和设计结果，为'{MEETING_TOPIC}'创建一份完整的项目方案:
        1. 项目概述
        2. 市场分析摘要
        3. 产品功能和特性
        4. 技术实现路线
        5. 用户体验设计要点
        6. 项目资源需求
        7. 风险评估
        8. 实施计划

        这是最终的会议成果，需要综合前面所有的分析和设计结果。
        输出一份完整的项目立项方案。
        """,
        agent=product_manager,
        expected_output="完整的项目立项方案文档",
        context=[market_analysis, product_definition, tech_solution, ux_design]
    )
    
    return [market_analysis, product_definition, tech_solution, ux_design, project_proposal]

# 主函数
def main():
    print("开始项目立项研究会议...")
    print(f"会议主题: {MEETING_TOPIC}\n")
    
    # 创建智能体和任务
    agents = create_agents()
    tasks = create_tasks(agents)
    
    # 创建会议团队
    project_crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,  # 将2改为True
        process=Process.sequential  # 会议流程是顺序的
    )
    
    # 开始会议
    result = project_crew.kickoff()
    
    # 获取最终任务的结果字符串
    final_result = ""
    if hasattr(result, "raw_output"):
        final_result = result.raw_output
    elif hasattr(result, "output"):
        final_result = result.output
    elif hasattr(result, "result"):
        final_result = result.result
    elif hasattr(result, "last_task_output"):
        final_result = result.last_task_output
    else:
        # 尝试直接转换为字符串
        final_result = str(result)
    
    # 输出会议结果
    print("\n\n最终项目方案:")
    print(final_result)
    
    # 保存会议结果到文件
    with open("meeting_result.md", "w", encoding="utf-8") as f:
        f.write(f"# {MEETING_TOPIC} - 项目立项方案\n\n")
        f.write(final_result)
        
    print("\n会议结果已保存到 meeting_result.md")

if __name__ == "__main__":
    main() 