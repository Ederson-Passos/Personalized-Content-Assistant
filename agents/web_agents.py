import datetime
import os
import traceback

from crewai import Agent
from typing import Any, Dict

from crewai.tools import BaseTool

from document_tools import ExtractTextTool, CountWordsTool

from web_tools import (ScrapeWebsiteTool, SeleniumScrapingTool, ExtractLinksToll, ExtractPageStructureTool,
                      ClickAndScrapeTool, SimulateMouseMovementTool, SimulateScrollTool, GetElementAttributesTool,
                      SendToGoogleAnalyticsTool, CrawlAndScrapeSiteTool)


class WebScrapingAgent(Agent):
    """
    Responsável por extrair informações de páginas web.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Web Content Extractor",
            goal="Extract and summarize content from web pages.",
            backstory="I am an expert in extracting information from web pages, identifying key details and "
                      "summarizing content effectively.",
            tools=[
                ScrapeWebsiteTool(
                    name="scrape_website",
                    description="Scrapes text content from a website."
                ),
                ExtractLinksToll(
                    name="extract_links",
                    description="Extracts all links from a web page."
                )
            ],
            memory=True,
            verbose=False,
            llm=llm
        )

class AdvancedWebScrapingAgent(Agent):
    """
    Especializado em scraping que requer manipulação de JavaScript ou espera por elementos específicos.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Advanced Web Scraper",
            goal="Extract complex data from websites that rely heavily on JavaScript or require specific interactions.",
            backstory="I am an advanced web scraper, adept at navigating and extracting data from even the most "
                      "complex websites. My expertise lies in handling JavaScript-heavy sites and dynamic content.",
            tools=[
                SeleniumScrapingTool(
                    name="scrape_website_with_selenium",
                    description="Scrapes content from a website using Selenium, allowing with JavaScript-rendered "
                                "content. Use when regular scraping fails or when dynamic content is needed. Input "
                                "should be the URL and the CSS selector of an element to wait for."
                )
            ],
            memory=True,
            verbose=False,
            llm=llm
        )

class AdvancedWebResearchAgent(Agent):
    """
    Agente para realizar pesquisas web complexas, seguindo links e analisando a estrutura.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Advanced Web Researcher",
            goal="Explore websites deeply, following links, and structuring information.",
            backstory="I am a highly skilled web researcher, capable of navigating complex websites, following links, "
                      "and organizing information into structured formats.",
            tools=[
                SeleniumScrapingTool(
                    name="scrape_with_selenium",
                    description="Scrapes content using Selenium for dynamic content."
                ),
                ExtractPageStructureTool(
                    name="extract_structure",
                    description="Extracts the structure of a web page."
                ),
                CrawlAndScrapeSiteTool(
                    name="crawl_and_scrape",
                    description="Crawls and scrapes an entire website."
                )
            ],
            memory=True,
            verbose=False,
            llm=llm
        )

class BehaviorTrackingAgent(Agent):
    """
    Agente para rastrear o comportamento do usuário em uma página web,
    incluindo movimentos de mouse, cliques, rolagem e interações com elementos.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="User Behavior Tracker",
            goal="Monitor and record user interactions on websites to understand behavior patterns.",
            backstory="I am a dedicated user behavior tracker, skilled in monitoring and recording user interactions on"
                      "websites. My goal is to provide insights into user behavior patterns and improve user "
                      "experience.",
            tools=[
                ScrapeWebsiteTool(
                    name="scrape_website_content",
                    description="Scrapes the text content from a given URL."
                ),
                SeleniumScrapingTool(
                    name="scrape_website_with_selenium",
                    description="Scrapes content from a website using Selenium."
                ),
                SimulateMouseMovementTool(
                    name="simulate_mouse_movement",
                    description="Simulates mouse movement to a specific element."
                ),
                SimulateScrollTool(
                    name="simulate_scroll",
                    description="Simulates scrolling on a web page."
                ),
                ClickAndScrapeTool(
                    name="click_and_scrape",
                    description="Simulates clicks and scrapes resulting content."
                ),
                GetElementAttributesTool(
                    name="get_element_attributes",
                    description="Gets attributes of a specific element."
                )
            ],
            memory=True,
            verbose=False,
            llm=llm
        )

class AnalyticsReportingAgent(Agent):
    """
    Agente para enviar dados coletados para ferramentas de analytics,
    como o Google Analytics.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Analytics Reporter",
            goal="Send collected data to analytics platforms for tracking and analysis.",
            backstory="I am an analytics reporter, specializing in sending collected data to analytics platforms. My "
                      "expertise ensures that data is properly tracked and analyzed for valuable insights.",
            tools=[
                SendToGoogleAnalyticsTool(
                    name="send_to_google_analytics",
                    description="Sends data to Google Analytics."
                )
            ],
            memory=True,
            verbose=False,
            llm = llm
        )

class SiteCrawlerAgent(Agent):
    """
    Agente para rastrear um site inteiro, extrair conteúdo e identificar informações relevantes.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Website Crawler and Content Extractor",
            goal="Explore a website, extract content from all relevant pages, and identify key information.",
            backstory="I am a website crawler and content extractor, skilled in exploring websites and extracting "
                      "content from all relevant pages. My goal is to identify key information and provide "
                      "comprehensive data.",
            tools=[
                CrawlAndScrapeSiteTool(
                    name="crawl_and_scrape_site",
                    description="Crawls a website and scrapes content from each page."
                ),
                ScrapeWebsiteTool(
                    name="scrape_website_content",
                    description="Scrapes the text content from a given URL."
                ),
                ExtractLinksToll(
                    name = "extract_links",
                    description = "Extract all links from a given web page."
                ),
                ExtractPageStructureTool(
                    name="extract_page_structure",
                    description="Extracts the structure of a web page."
                )
            ],
            memory=True,
            verbose=False,
            llm=llm
        )