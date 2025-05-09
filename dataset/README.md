# Spatial and POI Query Dataset

This repository contains a structured dataset designed to evaluate natural language understanding and generation capabilities of large language models used in **WalkRAG**, particularly in the context of spatial reasoning and information retrieval about points of interest (POIs). The dataset focuses on **Paris**.

## Contents
```
user_query.csv
```
A CSV file containing:
- 10 spatial request prompts: natural language request to go from one location to another.

- For each spatial request, 3 additional queries focused on retrieving general information about the POIs mentioned (e.g., history, visiting info, significance), totaling 30 POI-related queries.


**CSV Structure**:
```id```: Id of the request (e.g., R0 for spatial request 0, R0.1, R0.2, and R0.3 for general requests regarding R0)

```city```: Name of the city ("Paris" in our case)

```query```: Text of the query

```class```: The class indicating the type of the request (spatial or information)

```
instruction_llm.txt
```

A text file with the prompt instructions used to guide the LLM in processing both spatial and POI-related queries.

## Purpose

This dataset is intended for:

- Evaluating LLM capabilities in spatial reasoning and contextual POI information retrieval

- Studying prompt design and multi-query interactions with geographical context

