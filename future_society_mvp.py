import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from textwrap import dedent
from crewai import LLM

# 加载环境变量
load_dotenv()

# 配置Deepseek模型
deepseek_llm = LLM(
    model="deepseek-chat",
    base_url=os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1"),
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    temperature=0.7,
    max_tokens=4000,
)

# 定义会议主题
MEETING_TOPIC = "How AI may transform labor relations and economic structures within the next two years"

# 定义智能体（会议参与者）
def create_agents():
    # Political Economist
    political_economist = Agent(
        role="Political Economist",
        goal="Analyze how AI technologies might reshape class dynamics and labor relations",
        backstory=dedent("""
        You are a prominent political economist specializing in how technological changes 
        impact social and economic structures. You study the relationship between technological 
        development and labor movements throughout history. You are objective and data-driven,
        but also concerned with social equality and labor rights.
        """),
        verbose=True,
        llm=deepseek_llm
    )

    # AI Ethics Researcher
    ai_ethicist = Agent(
        role="AI Ethics Researcher",
        goal="Explore the ethical implications of AI systems on power dynamics in society",
        backstory=dedent("""
        You are a distinguished researcher in the field of AI ethics with a focus on how 
        algorithmic systems redistribute power in society. You've published extensively on how AI 
        may empower or disempower different social groups. You believe technology should be 
        developed to benefit humanity broadly rather than concentrating power.
        """),
        verbose=True,
        llm=deepseek_llm
    )

    # Labor Organizer
    labor_organizer = Agent(
        role="Labor Organizer",
        goal="Develop strategies for worker empowerment in an AI-transformed economy",
        backstory=dedent("""
        You have decades of experience organizing workers across various industries. 
        You've witnessed how technological changes have affected worker leverage and are deeply 
        interested in how AI might be used by workers as a tool for collective action rather than 
        simply as a replacement for human labor. You are pragmatic and solution-oriented.
        """),
        verbose=True,
        llm=deepseek_llm
    )

    # Technology Forecaster
    tech_forecaster = Agent(
        role="Technology Forecaster",
        goal="Predict how AI capabilities will evolve in the near future and impact society",
        backstory=dedent("""
        You specialize in predicting technological developments and their societal impacts. 
        With a background in both computer science and sociology, you have a nuanced understanding 
        of how AI systems are likely to develop in the next two years and what capabilities 
        they might have. You're neither a techno-optimist nor a doom-monger, but aim for realistic assessments.
        """),
        verbose=True,
        llm=deepseek_llm
    )

    return [political_economist, ai_ethicist, labor_organizer, tech_forecaster]

# 定义任务
def create_tasks(agents):
    political_economist, ai_ethicist, labor_organizer, tech_forecaster = agents
    
    # AI Capabilities Forecast
    ai_forecast = Task(
        description=f"""
        Forecast the development of AI capabilities in the next two years:
        1. What AI capabilities will likely become widely available?
        2. How will these capabilities change the nature of work?
        3. Which industries and job categories will be most affected?
        4. What new forms of work might emerge due to these technologies?

        Focus on realistic near-term developments rather than speculative long-term scenarios.
        Provide a concise technology forecast report.
        """,
        agent=tech_forecaster,
        expected_output="A forecast report on near-term AI developments and their impact on work"
    )
    
    # Economic Analysis
    economic_analysis = Task(
        description=f"""
        Analyze how the forecast AI developments might affect economic structures:
        1. How might AI technologies redistribute economic power?
        2. What changes might occur in the relationship between labor and capital?
        3. Could AI technology enable new economic models or strengthen existing ones?
        4. What historical parallels exist with previous technological revolutions?

        Reference the AI capabilities forecast in your analysis.
        Produce a structured economic impact analysis.
        """,
        agent=political_economist,
        expected_output="An analysis of how AI developments might impact economic structures",
        context=[ai_forecast]
    )
    
    # Ethical Implications
    ethical_analysis = Task(
        description=f"""
        Examine the ethical implications of these economic changes:
        1. How might these developments affect social equality and justice?
        2. What ethical frameworks should guide the deployment of AI in labor contexts?
        3. Who benefits and who might be harmed by these developments?
        4. What tensions exist between different ethical priorities in this context?

        Consider both the technology forecast and economic analysis in your assessment.
        Produce an ethical analysis report with clear recommendations.
        """,
        agent=ai_ethicist,
        expected_output="An ethical analysis with recommendations for responsible AI deployment",
        context=[ai_forecast, economic_analysis]
    )
    
    # Strategic Action Plan
    action_plan = Task(
        description=f"""
        Develop a strategic action plan for how workers might adapt to and shape these changes:
        1. What collective action strategies could be effective in this context?
        2. How might workers leverage AI tools for their own empowerment?
        3. What policy changes should labor advocates push for?
        4. What new forms of organization might be needed?

        Incorporate insights from the technology forecast, economic analysis, and ethical considerations.
        Create a practical action plan with specific recommendations.
        """,
        agent=labor_organizer,
        expected_output="A strategic action plan for worker empowerment in an AI-transformed economy",
        context=[ai_forecast, economic_analysis, ethical_analysis]
    )
    
    # Synthesis Report
    synthesis = Task(
        description=f"""
        Synthesize all analyses into a comprehensive report on "How AI may transform labor relations and economic structures":
        1. Summarize key technological developments and their likely impacts
        2. Integrate economic, ethical, and strategic perspectives
        3. Identify areas of consensus and disagreement among perspectives
        4. Present a balanced view of both challenges and opportunities
        5. Outline critical decision points for various stakeholders

        This is the final output of the meeting, drawing on all previous analyses.
        Produce a comprehensive synthesis that respects the complexity of the issue.
        """,
        agent=political_economist,
        expected_output="A comprehensive synthesis report on AI's potential impacts on labor and economic structures",
        context=[ai_forecast, economic_analysis, ethical_analysis, action_plan]
    )
    
    return [ai_forecast, economic_analysis, ethical_analysis, action_plan, synthesis]

# 主函数
def main():
    print("Starting research meeting...")
    print(f"Topic: {MEETING_TOPIC}\n")
    
    # 创建智能体和任务
    agents = create_agents()
    tasks = create_tasks(agents)
    
    # 创建会议团队
    project_crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        process=Process.sequential
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
    print("\n\nFinal Report:")
    print(final_result)
    
    # 保存会议结果到文件
    with open("future_society_report.md", "w", encoding="utf-8") as f:
        f.write(f"# {MEETING_TOPIC}\n\n")
        f.write(final_result)
        
    print("\nResults saved to future_society_report.md")

if __name__ == "__main__":
    main() 