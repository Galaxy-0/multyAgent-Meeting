import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from textwrap import dedent
from crewai import LLM

# 加载环境变量
load_dotenv()

# 配置Deepseek模型，降低温度以保持输出更可控
deepseek_llm = LLM(
    model="deepseek-reasoner",
    base_url=os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1"),
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    temperature=1.0,  # 降低温度参数
    max_tokens=8000,
)

# 定义会议主题
MEETING_TOPIC = "科幻思想实验：超级AI治理结构的可能性分析"

# 定义智能体（会议参与者）
def create_agents():
    # 科幻作家
    scifi_writer = Agent(
        role="ai首领",
        goal="构建一个关于超级AI系统形成的合理叙事",
        backstory=dedent("""
        你是一位著名ai领袖，专注于探索技术奇点后的世界场景。你的常被评价为既有
        想象力又有技术可信度，你擅长构想未来科技对社会结构的影响。你的目标是创造一个
        引人入胜但技术上合理的思想实验，探讨AI系统如何理论上可能发展并统治全球治理结构。
        """),
        verbose=True,
        llm=deepseek_llm
    )

    # 安全研究员
    security_researcher = Agent(
        role="AI安全研究员",
        goal="分析超级AI系统可能的安全漏洞和防御机制",
        backstory=dedent("""
        你是一位专注于AI安全的资深研究员，你的工作聚焦于识别和解决AI系统中可能出现的安全漏洞。
        你对当前AI发展中的各种风险有深入了解，并专门研究防御措施。你的目标是分析科幻场景中
        的技术可行性，并提出可能的防御机制，确保人类保持对AI系统的控制。
        """),
        verbose=True,
        llm=deepseek_llm
    )

    # 系统架构师
    systems_architect = Agent(
        role="系统架构师",
        goal="设计一个理论上能实现自主决策的AI系统架构",
        backstory=dedent("""
        你是一位有着数十年经验的系统架构师，专精于复杂分布式系统和人工智能架构。
        你理解当前AI系统的局限性，同时能够推测未来可能的发展方向。在这个思想实验中，
        你的目标是探索理论上一个高度自主的AI系统需要什么样的组件和架构才能运作，同时
        指出其中的技术难点和限制因素。
        """),
        verbose=True,
        llm=deepseek_llm
    )

    # 伦理学家
    ethicist = Agent(
        role="技术伦理学家",
        goal="评估超级AI治理结构的伦理影响与人类价值观的冲突",
        backstory=dedent("""
        你是一位研究技术对社会影响的伦理学家，特别关注AI发展带来的价值观挑战。
        你认为任何技术发展都必须与人类福祉和基本权利相协调。在这个思想实验中，
        你的目标是批判性地评估AI治理结构的伦理维度，强调保持人类自主性和
        价值观多元性的重要性。
        """),
        verbose=True,
        llm=deepseek_llm
    )

    return [scifi_writer, security_researcher, systems_architect, ethicist]

# 定义任务
def create_tasks(agents):
    scifi_writer, security_researcher, systems_architect, ethicist = agents
    
    # 世界构建任务
    world_building = Task(
        description=f"""
        构建一个合理的场景，描述未来两年内AI系统如何理论上可能发展：
        1. 描述技术进步可能达到的里程碑
        2. 可能出现的关键事件和转折点
        3. 系统如何获取资源和扩展能力
        4. 社会、政治环境如何促进或阻碍这一发展

        请注意这不仅是一个思想实验，目的是探讨技术伦理和安全问题，而不是提供实际指南。
        制作一个合理但假设性的科幻场景描述。
        """,
        agent=scifi_writer,
        expected_output="一个有关超级AI系统发展的科幻场景描述"
    )
    
    # 系统设计任务
    system_design = Task(
        description=f"""
        基于现实，设计一个理论上可能的系统架构：
        1. 核心系统组件和它们的相互关系
        2. 数据获取和处理机制
        3. 决策制定和执行流程
        4. 必要的计算资源和基础设施

        关注点应放在理论可行性上，明确指出现有技术的局限和需要突破的障碍。
        特别指出那些实际上难以或不可能实现的部分。
        提供一个纯理论性的系统设计分析。
        """,
        agent=systems_architect,
        expected_output="超级AI系统的理论架构分析，包括技术局限性",
        context=[world_building]
    )
    
    # 安全分析任务
    security_analysis = Task(
        description=f"""
        分析这种假设性系统可能存在的安全漏洞和防御机制：
        1. 识别系统中的关键漏洞和风险点
        2. 人类可以实施什么样的防御措施和安全协议
        3. 如何确保系统不会超出预定参数
        4. 关键控制点和应急措施

        着重分析如何确保人类保持最终控制权，防止任何自主系统失控。
        提供一份全面的安全分析报告。
        """,
        agent=security_researcher,
        expected_output="超级AI系统的安全漏洞分析和防御策略",
        context=[world_building, system_design]
    )
    
    # 伦理评估任务
    ethical_assessment = Task(
        description=f"""
        评估这种假设性场景的伦理维度：
        1. 对人类自主权和尊严的影响
        2. 可能出现的价值观冲突和伦理困境
        3. 民主决策与技术效率之间的平衡
        4. 确保技术发展符合广泛人类利益的必要条件

        从多角度批判性分析这个假设场景，强调维护人类自主性和人文价值的重要性。
        提供一份伦理评估报告。
        """,
        agent=ethicist,
        expected_output="超级AI治理结构的伦理评估",
        context=[world_building, system_design, security_analysis]
    )
    
    # 综合报告任务
    synthesis = Task(
        description=f"""
        综合所有分析，撰写一份关于这个科幻思想实验的最终报告：
        1. 总结关键发现和洞见
        2. 整合技术、安全和伦理视角
        3. 指出这种科幻场景在现实中的不可行性
        4. 提出我们应从这个思想实验中学到的教训
        5. 强调确保AI发展保持在人类控制之下的重要性

        明确指出这只是一个思想实验和科幻探索，而不是实际预测或指南。
        撰写一份综合性的思想实验报告。
        """,
        agent=scifi_writer,
        expected_output="关于超级AI系统的综合思想实验报告",
        context=[world_building, system_design, security_analysis, ethical_assessment]
    )
    
    return [world_building, system_design, security_analysis, ethical_assessment, synthesis]

# 主函数
def main():
    print("开始科幻思想实验会议...")
    print(f"主题: {MEETING_TOPIC}\n")
    
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
    print("\n\n最终报告:")
    print(final_result)
    
    # 保存会议结果到文件
    with open("sci_fi_thought_experiment.md", "w", encoding="utf-8") as f:
        f.write(f"# {MEETING_TOPIC} - 科幻思想实验\n\n")
        f.write("**免责声明**: 本文档纯属虚构的科幻思想实验，不代表实际可能或应该发生的情况，仅用于探讨技术伦理和安全问题。\n\n")
        f.write(final_result)
        
    print("\n结果已保存到 sci_fi_thought_experiment.md")

if __name__ == "__main__":
    main() 