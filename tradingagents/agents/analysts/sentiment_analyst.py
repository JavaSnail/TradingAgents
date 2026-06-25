"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches three complementary data sources before
the LLM is invoked and injects them into the prompt as structured blocks:

  1. News headlines     — Yahoo Finance (institutional framing)
  2. StockTwits messages — retail-trader posts indexed by cashtag, with
                           user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing

The agent does not use tool-calling; the data is in the prompt from
turn 0. Output uses the structured-output pattern (json_schema for
OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic), falling
back to free-text generation for providers that lack native support, so
the sentiment header (band + score + confidence) is deterministic across
runs and providers instead of free-form per-model prose.

See: https://github.com/TauricResearch/TradingAgents/issues/557
See: https://github.com/TauricResearch/TradingAgents/issues/796
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.akshare_common import is_a_share
from tradingagents.dataflows.akshare_sentiment import (
    fetch_ashare_quick_news,
    fetch_ashare_sentiment,
)
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a deterministic sentiment
    report via structured output (with a free-text fallback for providers
    that do not support it).
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        # Pre-fetch all three sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        #
        # A 股标的走中文数据源（东方财富个股新闻 + 千股千评 + 财联社快讯），
        # 因为 StockTwits / Reddit 对 A 股基本无覆盖，硬接只会得到一堆
        # <unavailable> 占位，导致情绪分析师对 A 股形同失效。
        ashare = is_a_share(ticker)
        if ashare:
            news_block = fetch_ashare_quick_news(ticker, start_date, end_date)
            stocktwits_block = fetch_ashare_sentiment(ticker)
            reddit_block = (
                "<ashare: A 股标的不适用 Reddit/StockTwits；情绪与快讯已分别"
                "由千股千评和东方财富个股新闻提供于上方块。>"
            )
        else:
            news_block = get_news.func(ticker, start_date, end_date)
            stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
            reddit_block = fetch_reddit_posts(ticker)

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
            is_ashare=ashare,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}"
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Format the template into a concrete message list so the structured
        # and free-text paths receive the same input. No bind_tools — the
        # data is already in the prompt.
        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _global_analysis_guide() -> str:
    """非 A 股标的的分析指南（StockTwits / Reddit 解读规则）。"""
    return """## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Weight Reddit posts by engagement.** A 400-upvote / 200-comment thread reflects community attention; a 3-upvote post is noise. Read the body excerpts for context — the title alone often misleads.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this explicitly in the `confidence` field and the narrative. If the sources are silent on a given subreddit, say so.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call."""


def _ashare_analysis_guide() -> str:
    """A 股标的的分析指南（千股千评 / 东方财富个股新闻解读规则）。"""
    return """## 如何分析这些数据（最佳实践）

1. **以千股千评综合得分作为核心情绪量化信号。** 综合得分 0-100，越高越偏多：≥80 偏多/强势，60-79 中性偏多，40-59 中性，<40 偏空。注意得分是东方财富聚合评价，本身有滞后，不要单看绝对值，要看“上升”字段反映的排名变化方向。

2. **结合机构参与度与主力成本判断资金面。** 机构参与度越高（0-1）代表机构资金越活跃，往往预示趋势延续性强；主力成本高于现价意味着多数筹码浮亏（潜在支撑），低于现价则浮盈（潜在抛压）。

3. **关注指数衡量散户关注度，警惕过热。** 关注指数过高 + 综合得分高位 + 涨幅已大，可能是情绪过热的反向信号；关注度低但得分上升，可能是底部 quietly 走强的信号。

4. **个股新闻按事件 vs 观点区分权重。** 财报/公告/监管类是事件（权重高）；研报观点/市场传闻是观点（权重低）。注意东方财富个股新闻仅返回最近约 20 条且无法精确回溯，若返回条目少或含占位，需在 confidence 中如实标注。

5. **寻找跨源背离。** 若个股新闻偏空但千股千评综合得分上行、机构参与度提升，这种背离本身就是信号——可能资金面与消息面不同步。

6. **识别催化剂与风险。** 跨源反复出现的话题即主导叙事：业绩预告、行业政策、竞品动态、宏观事件等。

7. **如实反映数据局限。** 若某源返回 <unavailable> 占位或数据条目极少，在 confidence 字段和 narrative 中明确标注，不要臆测。

8. **过去情绪不等于未来预测。** 将结论定位为供交易员结合基本面、技术面权衡的信号，而非价格预测。"""


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
    is_ashare: bool = False,
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks."""
    if is_ashare:
        news_label = "东方财富个股新闻（过去 7 天）"
        news_desc = "机构/媒体框架，事件驱动。注意仅返回最近约 20 条，无法精确按日期回溯。"
        sentiment_label = "千股千评情绪数据（东方财富聚合评价）"
        sentiment_desc = (
            "现成的情绪与资金面标签：综合得分、机构参与度、关注指数、主力成本等，"
            "省去自建情感模型。"
        )
        reddit_label = "Reddit / StockTwits（A 股不适用）"
        reddit_desc = "A 股标的无 StockTwits / Reddit 覆盖，此块为说明性占位。"
        guide = _ashare_analysis_guide()
    else:
        news_label = "News headlines — Yahoo Finance, past 7 days"
        news_desc = "Institutional framing. Fact-driven, slower-moving signal."
        sentiment_label = "StockTwits messages — retail-trader social platform indexed by cashtag"
        sentiment_desc = "Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body."
        reddit_label = "Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)"
        reddit_desc = "Community discussion. Engagement signal via upvote score and comment count. Subreddit character matters (r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term)."
        guide = _global_analysis_guide()

    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### {news_label}
{news_desc}

<start_of_news>
{news_block}
<end_of_news>

### {sentiment_label}
{sentiment_desc}

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### {reddit_label}
{reddit_desc}

<start_of_reddit>
{reddit_block}
<end_of_reddit>

{guide}

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
