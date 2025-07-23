import os

from .tools_and_schemas import SearchQueryList, ClassificationResult
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

from .state import (
    OverallState,
    QueryGenerationState,
    WebSearchState,
)
from .configuration import Configuration
from .prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    answer_instructions,
    classification_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from .utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Nodes
def classify_question(state: OverallState, config: RunnableConfig) -> dict:
    """
    Classifies the user's question to determine if it is related to law,
    notarization, or authentication.
    """
    # Always use Gemini 2.0 Flash for classification
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Gemini 2.0 Flash
        temperature=0,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(ClassificationResult)

    research_topic = get_research_topic(state["messages"])
    formatted_prompt = classification_instructions.format(
        research_topic=research_topic
    )

    result = structured_llm.invoke(formatted_prompt)

    return {"is_legal_question": result.is_legal_question}


def handle_non_legal_question(state: OverallState) -> dict:
    """
    Generates a response for questions that are not related to legal matters.
    """
    return {
        "messages": [
            AIMessage(
                content="Tôi xin lỗi, tôi chỉ có thể trả lời các câu hỏi liên quan đến luật, công chứng và chứng thực. Vui lòng đặt một câu hỏi khác."
            )
        ]
    }


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    
    # Check if grounding metadata exists and is not None
    if (response.candidates and 
        response.candidates[0] and 
        hasattr(response.candidates[0], 'grounding_metadata') and 
        response.candidates[0].grounding_metadata and 
        hasattr(response.candidates[0].grounding_metadata, 'grounding_chunks') and 
        response.candidates[0].grounding_metadata.grounding_chunks):
        
        # resolve the urls to short urls for saving tokens and time
        resolved_urls = resolve_urls(
            response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
        )
        # Gets the citations and adds them to the generated text
        citations = get_citations(response, resolved_urls)
        modified_text = insert_citation_markers(response.text, citations)
        sources_gathered = [item for citation in citations for item in citation["segments"]]
    else:
        # Handle case where no grounding metadata is available
        modified_text = response.text
        sources_gathered = []

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model, default to Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


def decide_what_to_do(state: OverallState) -> str:
    """
    Decides the next step based on the classification.
    """
    if state.get("is_legal_question"):
        return "generate_query"
    else:
        return "handle_non_legal_question"


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("classify_question", classify_question)
builder.add_node("handle_non_legal_question", handle_non_legal_question)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `classify_question`
# This means that this node is the first one called
builder.add_edge(START, "classify_question")

# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "classify_question",
    decide_what_to_do,
    {"generate_query": "generate_query", "handle_non_legal_question": "handle_non_legal_question"},
)
builder.add_edge("handle_non_legal_question", END)
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "finalize_answer")
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent") 