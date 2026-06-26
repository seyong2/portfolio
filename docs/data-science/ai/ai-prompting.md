---
title: My Journey Into Effective AI Prompting
parent: Artificial Intelligence
nav_order: 1
layout: default
---

Like most people, I’ve been using generative AI tools for a while now to help with daily tasks. But recently, I had a frustrating realization: I was using a supercomputer as if it were a basic search engine. I felt like I was barely scratching the surface of what these tools could actually do for my professional and personal life. To bridge that gap and finally learn how to maximize AI's capabilities, I decided to go back to school—metaphorically speaking—and took DeepLearning.AI's short course on Prompt Engineering. Here is what I learned, and how you can stop treating AI like a magic 8-ball and start treating it like a collaborative partner.

## The "Internet Frequency" Rule of Thumb

One of the most eye-opening early lessons from the course was a simple rule of thumb for measuring AI reliability: **An LLM’s reliability is directly tied to how frequently its training data was exposed to a topic on the internet**.
Think of pretrained knowledge as a reflection of global internet patterns. Because cooking, movies, and celebrities are universal, the AI has ingested billions of pages on them, making its responses highly reliable. But shift the topic to niche astrophysics (like quasars), and the reliability drops because the data pool is dramatically smaller. Furthermore, because the internet is heavily English-centric, the AI's reasoning is naturally sharpest in English. And, of course, it knows absolutely nothing about your company's private, proprietary data. If you want to know how much to trust an AI's output, ask yourself: How much public internet noise exists on this topic?

## Breaking the Time Barrier (How to Trigger Web Search)

An AI's static brain is frozen at its "knowledge cutoff date." It knows nothing about what happened after that date unless we trigger a web search. Fortunately, modern AI models can dynamically gather real-time data to bridge this gap.

How does this happen? Usually, in one of two ways:

1. The AI decides on its own: When you ask a question containing temporal cues (like "What is the 6 7 meme from 2025?"), the AI recognizes that "2025" is beyond its frozen training cutoff and automatically triggers a search. It does the same for location-specific queries or niche topics where its internal confidence is low.

2. You force it explicitly: You can manually trigger a search by clicking a search toggle in the UI, or by explicitly typing instructions like "search the web for..." in your prompt.

If your query involves timeless, common knowledge, the AI will rely purely on its pretrained memory. But for current events, real-time updates, or hyper-local details, make sure you are triggering that live search.

## Under the Hood of AI Web Search (And When to Use It)

When an AI searches the web, it doesn't actually look at Google the way we do. It uses a two-model architecture:

1. The User-Facing Model: This is the AI you chat with directly.

2. The Assistant Model: A backend assistant triggered to run web queries, filter out noise, download the most relevant pages, and summarize them for the user-facing model.

Because the user-facing model only reads summaries created by the assistant model, things can get lost in translation. This "summary bottleneck" is why AI sometimes cites a source that doesn't actually back up its claims when you click the link yourself.

Furthermore, web searches tend to bias toward high-volume public data (blogs, forums, social media) rather than verified scientific research. If you don't explicitly steer the AI (e.g., "Search only peer-reviewed academic sources"), it will default to the loudest voices on the web.

The Decision Matrix: Google vs. AI

- Use a traditional Search Engine (like Google) when: You need to navigate to a specific website, view raw data in its original format, or quickly scan multiple sources yourself.

- Use an AI Web Search when: You want a synthesized overview, need to weigh complex pros and cons, or want to contrast perspectives to form a thoughtful conclusion.

## Going Deep: The Power of "Deep Research" and Agentic Loops

When you need a highly rigorous, synthesized answer across dozens of sources, a standard web search won't cut it. This is where Deep Research comes in—a feature now native to major tools like ChatGPT, Gemini, and Claude.

Deep Research relies on agentic AI, meaning the model behaves like an independent agent that can make its own decisions about what to do next. Under the hood, this process follows a highly efficient, multi-step loop:

1. **The Research Plan**: When you input a complex query, the AI designs a structured research plan mapping out exactly what angles and source types it needs to explore. Most platforms allow you to review and edit this plan before launching it.

2. **Parallel Searching (Efficiency)**: Unlike a human who opens tabs one by one, the AI can execute dozens of different web searches simultaneously. This allows it to fetch massive amounts of web pages in seconds.

3. **The Agentic Loop (Evaluation & Course Correction)**: The AI reads the fetched pages, evaluates their relevance, and decides if it has enough information. If it finds gaps or new leads, it automatically generates new search terms and runs another round of searches.

4. **Synthesis & Report**: Once the AI determines it has sufficiently answered the query, it closes the loop. It synthesizes all the source pages into a comprehensive, heavily cited report.

How to use it: Deep Research is an active process that you usually trigger manually in the chat interface when you are tackling deep, multi-perspective investigations.

## Overcoming the "Average" Output: AI as a Brainstorming Partner

AI can be an exceptional partner for thinking through complex problems and decisions, especially because it excels at generating options. In brainstorming, we often say "the more ideas, the better," and AI can instantly hand you dozens of directions to choose from.

However, because AI is trained on public internet text, its default responses lean toward "common sense" and "the internet average." If you ask it a basic question, its slight mathematical randomness will give you slightly different variations of the same safe, predictable answers.

To bypass this "average" output and unlock high-quality, creative ideas, you must use an iterative feedback loop:

1. **Provide Context Up Front**: Start by sharing as much relevant detail and constraints about your problem as possible.

2. **Request a Small Batch**: Ask for 3 to 5 distinct options initially.

3. **Deliver Feedback**: Tell the AI exactly what you like and dislike about its suggestions. This feedback acts as critical new context for the model.

4. **Repeat the Loop**: Ask it to generate new options based on your critique.

By repeating this cycle several times, you guide the AI past its default answers into highly customized, innovative territory. Once you find a concept you love, you can then ask the AI to flesh out the specific details of that single idea.
