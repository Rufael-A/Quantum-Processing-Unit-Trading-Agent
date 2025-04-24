#!/usr/bin/env python3
"""
Quantum Trading Agent Project - Research Paper Collection
This script uses the Semantic Scholar API to search for relevant papers on quantum computing
for trading applications.
"""

import json
import time
import requests
from semanticscholar import SemanticScholar

# Initialize Semantic Scholar API with the provided key
API_KEY = "7dpJXWIakj7RpYNQVKqV47S3dSTb2oTD4GQjqb8O"
sch = SemanticScholar(api_key=API_KEY)

# Define search queries relevant to quantum trading
SEARCH_QUERIES = [
    "quantum computing finance",
    "quantum trading algorithm",
    "quantum portfolio optimization",
    "quantum machine learning finance",
    "quantum risk management",
    "quantum monte carlo finance",
    "quantum amplitude estimation finance",
    "quantum option pricing",
    "quantum asset pricing",
    "quantum reinforcement learning trading",
    "quantum computing sharpe ratio",
    "quantum computing market prediction",
    "quantum computing time series forecasting",
    "quantum computing high frequency trading",
    "IBM quantum finance"
]

# Function to search papers and save results
def search_papers(query, max_papers=10):
    """Search for papers using the Semantic Scholar API and return results."""
    print(f"Searching for: {query}")
    try:
        # Search for papers with the query
        papers = sch.search_paper(query, limit=max_papers, fields=[
            'title', 'abstract', 'year', 'authors', 'venue', 'citationCount', 
            'influentialCitationCount', 'url', 'externalIds'
        ])
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
        return papers
    except Exception as e:
        print(f"Error searching for '{query}': {e}")
        return []

# Function to get paper details
def get_paper_details(paper_id):
    """Get detailed information about a specific paper."""
    try:
        paper = sch.get_paper(paper_id, fields=[
            'title', 'abstract', 'year', 'authors', 'venue', 'citationCount',
            'influentialCitationCount', 'references', 'citations', 'url', 'externalIds'
        ])
        # Add a small delay to avoid rate limiting
        time.sleep(1)
        return paper
    except Exception as e:
        print(f"Error getting details for paper {paper_id}: {e}")
        return None

# Main function to collect papers
def collect_papers():
    """Collect papers from all search queries and save to JSON file."""
    all_papers = []
    paper_ids = set()  # To track unique papers
    
    # Search for papers using each query
    for query in SEARCH_QUERIES:
        papers = search_papers(query)
        
        # Add unique papers to the collection
        for paper in papers:
            # Access attributes directly from Paper object
            paper_id = paper.paperId
            if paper_id and paper_id not in paper_ids:
                paper_ids.add(paper_id)
                # Convert Paper object to dictionary for JSON serialization
                paper_dict = {
                    'paperId': paper.paperId,
                    'title': paper.title,
                    'abstract': paper.abstract,
                    'year': paper.year,
                    'authors': [{'name': author.name} for author in paper.authors] if paper.authors else [],
                    'venue': paper.venue,
                    'citationCount': paper.citationCount,
                    'url': paper.url
                }
                all_papers.append(paper_dict)
    
    print(f"Found {len(all_papers)} unique papers.")
    
    # Get more details for the top papers (by citation count)
    all_papers.sort(key=lambda x: x.get('citationCount', 0), reverse=True)
    detailed_papers = []
    
    # Get details for top 30 papers (to ensure we have at least 20 good ones)
    for paper in all_papers[:30]:
        paper_id = paper.get('paperId')
        if paper_id:
            detailed_paper = get_paper_details(paper_id)
            if detailed_paper:
                # Convert Paper object to dictionary for JSON serialization
                paper_dict = {
                    'paperId': detailed_paper.paperId,
                    'title': detailed_paper.title,
                    'abstract': detailed_paper.abstract,
                    'year': detailed_paper.year,
                    'authors': [{'name': author.name} for author in detailed_paper.authors] if detailed_paper.authors else [],
                    'venue': detailed_paper.venue,
                    'citationCount': detailed_paper.citationCount,
                    'url': detailed_paper.url,
                    'references': [{'paperId': ref.paperId, 'title': ref.title} 
                                  for ref in detailed_paper.references] if detailed_paper.references else []
                }
                detailed_papers.append(paper_dict)
    
    # Save all papers to a JSON file
    with open('all_papers.json', 'w') as f:
        json.dump(all_papers, f, indent=2)
    
    # Save detailed papers to a JSON file
    with open('detailed_papers.json', 'w') as f:
        json.dump(detailed_papers, f, indent=2)
    
    # Create a summary file with key information
    create_summary(detailed_papers)
    
    return all_papers, detailed_papers

def create_summary(papers):
    """Create a summary of the collected papers."""
    with open('paper_summary.md', 'w') as f:
        f.write("# Quantum Computing for Finance - Research Papers Summary\n\n")
        
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'No Title')
            year = paper.get('year', 'N/A')
            authors = ', '.join([author.get('name', 'Unknown') for author in paper.get('authors', [])])
            venue = paper.get('venue', 'N/A')
            citations = paper.get('citationCount', 0)
            abstract = paper.get('abstract', 'No abstract available.')
            url = paper.get('url', '')
            
            f.write(f"## {i}. {title} ({year})\n\n")
            f.write(f"**Authors:** {authors}\n\n")
            f.write(f"**Venue:** {venue}\n\n")
            f.write(f"**Citations:** {citations}\n\n")
            f.write(f"**URL:** {url}\n\n")
            f.write("**Abstract:**\n\n")
            f.write(f"{abstract}\n\n")
            f.write("---\n\n")

if __name__ == "__main__":
    print("Starting paper collection...")
    all_papers, detailed_papers = collect_papers()
    print(f"Paper collection complete. Found {len(all_papers)} papers in total.")
    print(f"Detailed information collected for {len(detailed_papers)} papers.")
    print("Results saved to all_papers.json, detailed_papers.json, and paper_summary.md")
