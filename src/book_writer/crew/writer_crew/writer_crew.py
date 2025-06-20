from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel
import os

llm = LLM( model="gpt-4o-mini-2024-07-18",  
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base="https://api.openai.com/v1")

class Chapter(BaseModel):
    """Chapter of the book"""
    title: str
    content: str

@CrewBase
class ChapterWriterCrew:
    """Chapter Writer Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def topic_researcher(self) -> Agent:
        return Agent(config=self.agents_config["topic_researcher"],
                     llm=llm)

    @task
    def research_topic(self) -> Task:
        return Task(config=self.tasks_config["research_topic"])

    @agent
    def writer(self) -> Agent:
        return Agent(config=self.agents_config["writer"],
                     llm=llm)

    @task
    def write_chapter(self) -> Task:
        return Task(config=self.tasks_config["write_chapter"],
                    output_pydantic=Chapter)

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""

        return Crew(agents=self.agents,
                    tasks=self.tasks,
                    process=Process.sequential,
                    verbose=True)