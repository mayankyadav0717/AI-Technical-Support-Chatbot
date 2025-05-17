"""
Stack Overflow Framework Questions and Answers Scraper

This script uses the Stack Exchange API to fetch framework-related questions
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
import configparser

class StackOverflowScraper:
    def __init__(self, tag="pytorch", output_dir=None):
        """Initialize the scraper with base URL and output directory."""
        self.base_url = "https://api.stackexchange.com/2.3"
        self.tag = tag.lower()
        
        # Create default output directory based on tag if none provided
        if output_dir is None:
            output_dir = f"{tag.lower()}_qa_data"
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load API key from config if available
        self.api_key = self._load_api_key()
    
    def _load_api_key(self):
        """Load API key from config file or environment variable."""
        # Try to load from config file
        config = configparser.ConfigParser()
        if os.path.exists('config.ini'):
            config.read('config.ini')
            if 'API' in config and 'key' in config['API']:
                return config['API']['key']
        
        # Try to load from environment variable
        return os.environ.get('STACKEXCHANGE_API_KEY', '')
    
    def fetch_questions(self, page=1, pagesize=100, sort="votes", max_retries=3):
        """Fetch questions with the specified tag, sorted by votes (score) by default."""
        endpoint = f"{self.base_url}/questions"
        
        params = {
            "page": page,
            "pagesize": pagesize,
            "order": "desc",  # Descending order (highest first)
            "sort": sort,     # Now using the provided sort parameter
            "tagged": self.tag,
            "site": "stackoverflow",
            "filter": "withbody"  # Include the question body
        }
        
        # Add API key if available
        if self.api_key:
            params["key"] = self.api_key
        
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(endpoint, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    # Check for backoff signal
                    if "backoff" in data:
                        backoff_time = int(data["backoff"]) + 1
                        print(f"API backoff requested: waiting {backoff_time} seconds")
                        time.sleep(backoff_time)
                    return data
                elif response.status_code == 429:  # Too Many Requests
                    wait_time = 30 * (retries + 1)  # Exponential backoff
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry ({retries+1}/{max_retries})...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    print(f"Error fetching questions: {response.status_code}")
                    print(f"Response content: {response.text[:200]}...")  # Print part of the response for debugging
                    return None
            except Exception as e:
                print(f"Exception while fetching questions: {str(e)}")
                if retries < max_retries - 1:
                    wait_time = 5 * (retries + 1)
                    print(f"Retrying in {wait_time} seconds... ({retries+1}/{max_retries})")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    return None
        
        print("Max retries reached. Could not complete request.")
        return None
    
    def fetch_answers(self, question_id, max_retries=3):
        """Fetch answers for a specific question."""
        endpoint = f"{self.base_url}/questions/{question_id}/answers"
        
        params = {
            "order": "desc",
            "sort": "votes",
            "site": "stackoverflow",
            "filter": "withbody"  # Include the answer body
        }
        
        # Add API key if available
        if self.api_key:
            params["key"] = self.api_key
        
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(endpoint, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    # Check for backoff signal
                    if "backoff" in data:
                        backoff_time = int(data["backoff"]) + 1
                        print(f"API backoff requested: waiting {backoff_time} seconds")
                        time.sleep(backoff_time)
                    return data
                elif response.status_code == 429:  # Too Many Requests
                    wait_time = 30 * (retries + 1)  # Exponential backoff
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry ({retries+1}/{max_retries})...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    print(f"Error fetching answers for question {question_id}: {response.status_code}")
                    print(f"Response content: {response.text[:200]}...")  # Print part of the response for debugging
                    return None
            except Exception as e:
                print(f"Exception while fetching answers: {str(e)}")
                if retries < max_retries - 1:
                    wait_time = 5 * (retries + 1)
                    print(f"Retrying in {wait_time} seconds... ({retries+1}/{max_retries})")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    return None
        
        print("Max retries reached. Could not complete request.")
        return None
    
    def process_data(self, total_questions=1000, questions_per_page=100, adaptive_rate_limit=True):
        """Process questions and their answers, storing them in a structured way.
        Only includes questions with at least one answer, and only keeps the highest-voted answer.
        
        Args:
            total_questions: Total number of highest-voted questions to scrape
            questions_per_page: Number of questions per API request (max 100 for Stack Exchange API)
            adaptive_rate_limit: Dynamically adjust delays based on API response
        """
        all_qa_data = []
        num_pages = (total_questions + questions_per_page - 1) // questions_per_page  # Ceiling division
        
        # Initial delays
        page_delay = 1.0  # Seconds between page requests
        answer_delay = 0.5  # Seconds between answer requests
        consecutive_429s = 0  # Counter for consecutive 429 errors
        
        for page in range(1, num_pages + 1):
            print(f"Fetching page {page} of {self.tag} questions... ({len(all_qa_data)}/{total_questions} collected so far)")
            questions_data = self.fetch_questions(page=page, pagesize=questions_per_page)
            
            # Check if we got valid data back
            if not questions_data:
                print(f"No data returned for page {page}, skipping to next page")
                # Increase delay if we're having issues
                if adaptive_rate_limit:
                    page_delay *= 1.5
                    print(f"Increasing page delay to {page_delay:.2f} seconds")
                continue
                
            if "items" in questions_data:
                # Reset consecutive 429s counter on successful request
                consecutive_429s = 0
                
                # Process each question
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
                    
                    if answers_data is None and adaptive_rate_limit:
                        # Likely hit rate limit, increase delay and count
                        consecutive_429s += 1
                        answer_delay *= 2
                        print(f"Increasing answer delay to {answer_delay:.2f} seconds")
                        
                        # If multiple consecutive failures, take a longer break
                        if consecutive_429s >= 3:
                            cooldown = 120  # 2 minute cooldown
                            print(f"Multiple consecutive failures. Taking a {cooldown} second break...")
                            time.sleep(cooldown)
                            consecutive_429s = 0
                        
                        continue
                    else:
                        # Reset counter on success
                        consecutive_429s = 0
                    
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
                    
                    # Respect API rate limits - adaptive delay
                    time.sleep(answer_delay)
            else:
                print(f"No 'items' in response data for page {page}")
            
            # If there are no more pages, break
            has_more = questions_data.get("has_more", False) if questions_data else False
            if not has_more:
                print("No more pages available")
                break
                
            # Add a delay between pages to respect API rate limits
            print(f"Waiting {page_delay:.2f} seconds before next page...")
            time.sleep(page_delay)
        
        return all_qa_data
    
    def save_data(self, data, format="json"):
        """Save the collected data to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{self.tag}_qa_{timestamp}.{format}"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Data saved to {filename}")
        return filename


def main():
    """Main function to run the scraper."""
    # List of ML/Dev frameworks to scrape
    frameworks = [
        "TensorFlow", "Keras", "HuggingFace", "CUDA", "PyTorch", 
        "NumPy", "Pandas", "Matplotlib", "Scikit-learn", "Docker", 
        "VSCode", "WandB", "TensorBoard", "OpenAI-Gym", "Langchain", 
        "Gradio", "Streamlit", "FastAPI"
    ]
    
    parser = argparse.ArgumentParser(description="Scrape framework questions and answers from Stack Overflow")
    parser.add_argument("--framework", type=str, choices=frameworks + [f.lower() for f in frameworks], 
                        default="pytorch", help="Framework to scrape questions about")
    parser.add_argument("--total", type=int, default=1000, help="Total number of highest-scored questions to scrape")
    parser.add_argument("--pagesize", type=int, default=100, help="Number of questions per page (max 100 for API)")
    parser.add_argument("--output", type=str, help="Output directory (default: [framework]_qa_data)")
    parser.add_argument("--all", action="store_true", help="Scrape all frameworks in the list")
    
    args = parser.parse_args()
    
    if args.all:
        # Scrape all frameworks
        for framework in frameworks:
            process_framework(framework, args.total, args.pagesize, args.output)
    else:
        # Scrape just the specified framework
        process_framework(args.framework, args.total, args.pagesize, args.output)


def process_framework(framework, total_questions, pagesize, output_dir=None, incremental=True):
    """Process a single framework and save its data."""
    # If output directory is provided, use it with framework subfolder
    if output_dir:
        framework_dir = os.path.join(output_dir, framework.lower())
    else:
        framework_dir = f"{framework.lower()}_qa_data"
    
    scraper = StackOverflowScraper(tag=framework, output_dir=framework_dir)
    print(f"Starting to scrape {total_questions} highest-voted {framework} questions from Stack Overflow...")
    
    # Check if we have existing data files for this framework
    existing_count = 0
    existing_data = []
    
    if incremental and os.path.exists(framework_dir):
        data_files = [f for f in os.listdir(framework_dir) if f.endswith('.json')]
        if data_files:
            # Find the most recent file
            data_files.sort(reverse=True)
            newest_file = os.path.join(framework_dir, data_files[0])
            print(f"Found existing data file: {newest_file}")
            
            try:
                with open(newest_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                existing_count = len(existing_data)
                print(f"Loaded {existing_count} existing records. Will collect {total_questions - existing_count} more.")
            except Exception as e:
                print(f"Error loading existing data: {str(e)}")
                existing_data = []
    
    try:
        # Adjust count to collect only what we need
        questions_to_get = max(0, total_questions - existing_count)
        
        if questions_to_get > 0:
            qa_data = scraper.process_data(total_questions=questions_to_get, questions_per_page=pagesize)
            
            if qa_data:
                # Combine with existing data if any
                combined_data = existing_data + qa_data
                filename = scraper.save_data(combined_data)
                print(f"Successfully scraped {len(qa_data)} new {framework} questions with their best answers.")
                print(f"Total dataset now contains {len(combined_data)} records.")
                print(f"Data saved to {filename}")
            else:
                print(f"No new data was scraped for {framework}.")
        else:
            print(f"Already have {existing_count} records for {framework}, which meets the target of {total_questions}.")
            
    except Exception as e:
        print(f"An error occurred during scraping {framework}: {str(e)}")
    
    print(f"Finished processing {framework}\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()