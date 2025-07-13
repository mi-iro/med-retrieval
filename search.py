import json
import requests

def process_and_retrieve_data(file_path, api_url):
    """
    Reads question and options from a JSON file, sends them to a retrieval API,
    and stores the results.

    Args:
        file_path (str): The path to the JSON file.
        api_url (str): The URL of the retrieval API.
    """
    # Load the JSON data from the file
    with open(file_path, 'r') as f:
        data = json.load(f)
    import random
    data = data[:500]
    random.shuffle(data)
    data = data[:20]
    all_results = []
    sum_score = 0

    # Iterate over each record in the JSON data
    for record in data:
        question = record.get("question", "")
        options = record.get("options", "")
        # print(question)

        # Concatenate the question and options to form the query
        query_text = f"{question}\n{options}"

        # Prepare the data for the POST request
        post_data = {
            "question": query_text,
            "topk": 10,  # As specified in the user's curl command
            "n_queries": 3,  # As specified in the user's curl command
            "rate": 10,
        }

        try:
            # Send the request to the API
            response = requests.post(api_url, json=post_data)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            retrieved_data = response.json()
            
            # Store the results
            all_results.append({
                "id": record.get("id"),
                "retrieved_data": retrieved_data
            })

            print(f"Successfully retrieved data for ID: {record.get('id')}")
            score = retrieved_data['performance_metrics']['average_rerank_score']
            sum_score += score
            print("Rerank Score:", score)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred for ID {record.get('id')}: {e}")

    # Save the results to a new JSON file
    with open('retrieval_results_'+file_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print("\nRetrieval process finished. Results saved to " + 'retrieval_results_'+file_path)
    print("Avg Rerank Score:", sum_score / len(data))

if __name__ == '__main__':
    # The user should replace 'test_gpqam_diamond10.json' with the actual path to their file
    # if it's not in the same directory as the script.
    json_file_path = 'test_mmluprom.json'
    retrieval_api_url = 'http://127.0.0.1:8888/process-query/'
    
    # # Create a dummy JSON file for demonstration purposes, as I cannot access local files.
    # # The user should have their own 'test_gpqam_diamond10.json' file.
    # dummy_data = [
    #   {
    #     "id": "171",
    #     "question": "You perform a high-throughput experiment on white lupine... Which conclusion regarding those genes' interaction can you draw from this experiment?\n",
    #     "options": "A.G2 is a transcription factor, G1 and G3 show pleiotropy, G1 is epistatic towards G3\nB.G2 is a transcription factor, G1 and G3 has the same promoter, G3 is epistatic towards G1\nC.G1 is a transcription factor, G2 and G3 show pleiotropy, G2 is epistatic towards G1\nD.G2 is a transcription factor, G1 and G3 show gene redundancy, G1 is epistatic towards G3",
    #     "output": "D"
    #   },
    #   {
    #       "id": "172",
    #       "question": "All the following statements about the molecular biology of Severe Acute Respiratory Syndrome Coronavirus 2 (SARS‑CoV‑2) are correct except\n\n\n",
    #       "options": "A.The rate of frameshifting in vitro is linearly correlated with the number of conformations that a pseudoknot can adopt... \nB.SARS-CoV-2 ORF3a has the ability to trigger caspase-8 activation/cleavage... \nC.SARS-CoV-2 nsp10/nsp14-ExoN operates as heterodimers in a mismatch repair mechanism... \nD.Programmed ribosomal frameshifting creates two polyproteins near to 5` end of the genome..."
    #   }
    # ]

    # with open(json_file_path, 'w') as f:
    #     json.dump(dummy_data, f, indent=4)

    process_and_retrieve_data(json_file_path, retrieval_api_url)