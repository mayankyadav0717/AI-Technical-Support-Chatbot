"""
Stack Overflow PyTorch Questions and Answers Scraper

This script uses the Stack Exchange API to fetch PyTorch-related questions
and their answers from Stack Overflow, then stores them in a structured JSON file.
Only questions with at least one answer are included, and only the highest-voted
answer is stored for each question.
"""

import requests
import json
import time
import os
from datetime import datetime
import argparse

class StackOverflowScraper:
    def __init__(self, output_dir="pytorch_qa_data"):
        """Initialize the scraper with base URL and output directory."""
        self.base_url = "https://api.stackexchange.com/2.3"
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def fetch_questions(self, tag="pytorch", page=1, pagesize=100, sort="votes"):
        """Fetch questions with the specified tag, sorted by votes (score) by default."""
        endpoint = f"{self.base_url}/questions"
        
        # Always use "votes" for sorting to get the highest scored questions first
        params = {
            "page": page,
            "pagesize": pagesize,
            "order": "desc",  # Descending order (highest first)
            "sort": "votes",  # Always sort by votes/score regardless of the sort parameter
            "tagged": tag,
            "site": "stackoverflow",
            "filter": "withbody",  # Include the question body
            "key": "U4DMV*8nvpm3EOpvf69Rxw(("  # Add API key to increase quota
        }
        
        try:
            response = requests.get(endpoint, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching questions: {response.status_code}")
                print(f"Response content: {response.text[:200]}...")  # Print part of the response for debugging
                return None
        except Exception as e:
            print(f"Exception while fetching questions: {str(e)}")
            return None
    
    def fetch_answers(self, question_id):
        """Fetch answers for a specific question."""
        endpoint = f"{self.base_url}/questions/{question_id}/answers"
        
        params = {
            "order": "desc",
            "sort": "votes",
            "site": "stackoverflow",
            "filter": "withbody",  # Include the answer body
            "key": "U4DMV*8nvpm3EOpvf69Rxw(("  # Add API key to increase quota
        }
        
        try:
            response = requests.get(endpoint, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching answers for question {question_id}: {response.status_code}")
                print(f"Response content: {response.text[:200]}...")  # Print part of the response for debugging
                return None
        except Exception as e:
            print(f"Exception while fetching answers: {str(e)}")
            return None
    
    def process_data(self, total_questions=1000, questions_per_page=100):
        """Process questions and their answers, storing them in a structured way.
        Only includes questions with at least one answer, and only keeps the highest-voted answer.
        
        Args:
            total_questions: Total number of highest-voted questions to scrape
            questions_per_page: Number of questions per API request (max 100 for Stack Exchange API)
        """
        all_qa_data = []
        num_pages = (total_questions + questions_per_page - 1) // questions_per_page  # Ceiling division
        
        for page in range(1, num_pages + 1):
            print(f"Fetching page {page} of questions... ({len(all_qa_data)}/{total_questions} collected so far)")
            questions_data = self.fetch_questions(page=page, pagesize=questions_per_page)
            
            # Check if we got valid data back
            if not questions_data:
                print(f"No data returned for page {page}, skipping to next page")
                continue
                
            if "items" in questions_data:
                for question in questions_data["items"]:
                    # Skip questions with no answers
                    if question["answer_count"] == 0:
                        print(f"Skipping question {question['question_id']} - no answers")
                        continue
                        
                    # Basic question information
                    question_id = question["question_id"]
                    qa_item = {
                        "question_id": question_id,
                        "title": question["title"],
                        "body": question["body"],
                        "score": question["score"],
                        "view_count": question.get("view_count", 0),
                        "tags": question["tags"],
                        "is_answered": question["is_answered"],
                        "creation_date": question["creation_date"],
                        "link": question["link"],
                        "best_answer": None
                    }
                    
                    # Fetch answers
                    print(f"Fetching answers for question {question_id}...")
                    answers_data = self.fetch_answers(question_id)
                    
                    # Find the highest-voted answer
                    if answers_data and "items" in answers_data and len(answers_data["items"]) > 0:
                        # Sort answers by score (highest first)
                        sorted_answers = sorted(answers_data["items"], key=lambda x: x["score"], reverse=True)
                        
                        # Get the highest-voted answer
                        best_answer = sorted_answers[0]
                        
                        # Check if there's an accepted answer with a reasonably high score
                        # (Sometimes an accepted answer might be better even with slightly fewer votes)
                        for answer in answers_data["items"]:
                            if answer.get("is_accepted", False) and answer["score"] >= best_answer["score"] * 0.8:
                                best_answer = answer
                                break
                                
                        # Add the best answer to our data
                        qa_item["best_answer"] = {
                            "answer_id": best_answer["answer_id"],
                            "body": best_answer["body"],
                            "score": best_answer["score"],
                            "is_accepted": best_answer.get("is_accepted", False),
                            "creation_date": best_answer["creation_date"]
                        }
                        
                        # Only add questions that have at least one answer
                        all_qa_data.append(qa_item)
                    
                    # Check if we've collected enough questions
                    if len(all_qa_data) >= total_questions:
                        print(f"Reached target of {total_questions} questions")
                        return all_qa_data
                    
                    # Respect API rate limits
                    time.sleep(0.1)
            else:
                print(f"No 'items' in response data for page {page}")
            
            # If there are no more pages, break
            has_more = questions_data.get("has_more", False) if questions_data else False
            if not has_more:
                print("No more pages available")
                break
                
            # Add a delay between pages to respect API rate limits
            time.sleep(0.5)
        
        return all_qa_data
    
    def save_data(self, data, format="json"):
        """Save the collected data to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/pytorch_qa_{timestamp}.{format}"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Data saved to {filename}")
        return filename

def main():
    """Main function to run the scraper."""
    frameworks = ["pytorch","TensorFlow","Keras","HugginFace","CUDA","Pytorch","Numpy","Pandas","Matplotlib","Scikit-learn","Docker","VSCode","WandB","TensorBoard","OpenAI Gym","Langchain","Gradio","Streamlit","FastAPI"]
    for framework in frameworks:
        parser = argparse.ArgumentParser(description=f"Scrape {framework} questions and answers from Stack Overflow")
        parser.add_argument("--total", type=int, default=1000, help="Total number of highest-scored questions to scrape")
        parser.add_argument("--pagesize", type=int, default=100, help="Number of questions per page (max 100 for API)")
        parser.add_argument("--output", type=str, default="pytorch_qa_data", help="Output directory")

        args = parser.parse_args()

        scraper = StackOverflowScraper(output_dir=args.output)
        print(f"Starting to scrape {args.total} highest-voted PyTorch questions from Stack Overflow...")

        try:
            qa_data = scraper.process_data(total_questions=args.total, questions_per_page=args.pagesize)

            if qa_data:
                filename = scraper.save_data(qa_data)
                print(f"Successfully scraped {len(qa_data)} questions with their best answers.")
                print(f"Data saved to {filename}")
            else:
                print("No data was scraped.")

        except Exception as e:
            print(f"An error occurred during scraping: {str(e)}")

if __name__ == "__main__":
    main()