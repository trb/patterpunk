
import json

from patterpunk.lib.extract_json import extract_json


def test_complex_json():
    json_str = """```json
{
  "title": "The Impact of Artificial Intelligence on Climate Change Mitigation",
  "summary": "This research paper examines the dual role of AI in climate change, analyzing how AI technologies can both contribute to greenhouse gas emissions through energy consumption and help mitigate climate change through improved energy management, climate modeling, and renewable energy optimization. The study includes case studies across five continents showing AI-optimized systems can reduce energy consumption by 15-30% in various sectors, while also addressing ethical concerns about unequal distribution of AI benefits and climate impacts.",
  "key_points": [
    "AI has both positive and negative impacts on climate change",
    "AI systems require significant energy for training and inference",
    "AI enables more efficient energy management and improved climate modeling",
    "AI-optimized systems can reduce energy consumption by 15-30% across various sectors",
    "There are ethical concerns regarding unequal distribution of AI benefits and climate impacts",
    "The paper provides policy recommendations for aligning AI development with climate goals"
  ],
  "sentiment": "neutral",
  "topics": [
    "artificial intelligence",
    "climate change",
    "energy efficiency",
    "environmental policy",
    "ethics",
    "renewable energy",
    "greenhouse gas emissions"
  ],
  "author": {
    "name": "Unknown",
    "expertise": ["artificial intelligence", "climate science", "environmental policy"],
    "years_experience": 5
  },
  "references": [],
  "confidence_score": 0.85,
  "is_factual": true
}
```"""

    jsons = extract_json(json_str)
    # @todo add assertions that the decoded json correctly replicates all the fields from the json_str. Shorten the fields in the json since the length doesn't matter and they're pretty long right now.
    
    # Check that we got a list with one JSON object
    assert len(jsons) == 1
    
    # Get the extracted JSON object
    extracted_json = json.loads(jsons[0])
    
    # Check all top-level fields exist
    assert "title" in extracted_json
    assert "summary" in extracted_json
    assert "key_points" in extracted_json
    assert "sentiment" in extracted_json
    assert "topics" in extracted_json
    assert "author" in extracted_json
    assert "references" in extracted_json
    assert "confidence_score" in extracted_json
    assert "is_factual" in extracted_json
    
    # Check field values
    assert extracted_json["title"] == "The Impact of Artificial Intelligence on Climate Change Mitigation"
    assert "AI in climate change" in extracted_json["summary"]
    
    # Check arrays
    assert len(extracted_json["key_points"]) == 6
    assert "AI has both positive and negative impacts on climate change" in extracted_json["key_points"]
    
    assert len(extracted_json["topics"]) == 7
    assert "artificial intelligence" in extracted_json["topics"]
    assert "climate change" in extracted_json["topics"]
    
    # Check nested object
    assert extracted_json["author"]["name"] == "Unknown"
    assert len(extracted_json["author"]["expertise"]) == 3
    assert "artificial intelligence" in extracted_json["author"]["expertise"]
    assert extracted_json["author"]["years_experience"] == 5
    
    # Check empty array
    assert extracted_json["references"] == []
    
    # Check numeric and boolean values
    assert extracted_json["confidence_score"] == 0.85
    assert extracted_json["is_factual"] is True
    
    # Check sentiment
    assert extracted_json["sentiment"] == "neutral"
