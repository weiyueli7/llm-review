import json
import re
from agents import LlamaAgent, MistralAgent, OpenAIAgent
import datetime
import os
from tqdm import tqdm
import numpy as np
import torch
import random
import multiprocessing
import io

class Discussion:
    PROMPTS = {
        1: "You are in a group discussion with other teammates; as a result, answer as diversely and creatively as you can.",
        2: "You're in a brainstorming session where each idea leads to the next. Embrace the flow of creativity without limits, encouraging one another to build on each suggestion for unexpected connections.",
        3: "Pretend your team is at a think tank where unconventional ideas are the norm. Challenge each other to think from different perspectives, considering the most unusual or innovative ideas.",
        4: "Engage in a collaborative discussion where each of you contributes a unique insight or query, aiming to delve into uncharted territories of thought. Throughout the discussion, focus on expanding the scope and depth of each contribution through constructive feedback, counterpoints, and further questioning. The objective is to achieve a broad spectrum of ideas and solutions, promoting a culture of continuous learning and innovation.",
        5: "Envision your group as a crew on a mission to solve a mystery using only your creativity and wit. How would you piece together clues from each member's ideas to find the solution? And this would be crucial to your member’s life."
    }

    def __init__(self, dataset_file, rounds, prompt, secrets_path="../../secrets.json"):
        self.dataset_file = dataset_file
        self.rounds = rounds
        self.discussion_prompt = self.PROMPTS.get(prompt, "Invalid prompt selected.")
        print("Discussion initialized with dataset: {} and {} rounds.".format(dataset_file, rounds))

    def save_conversation(self, filename, conversation_data):
        """Save conversation data to local file system."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save to local file
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=4)
        print(f"Saved Conversation Data to local file: {filename}")

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

        
    def extract_response(self, content):
        lines = content.split('\n')
        uses = [line.strip() for line in lines if line.strip() and re.match(r"^\d+\.", line)]
        uses = [use[use.find('.') + 2:] for use in uses]
        return uses

class LLM_Debate(Discussion):
    def __init__(self, agents_config, dataset_file, rounds, task, prompt):
        super().__init__(dataset_file, rounds, prompt)
        self.task_type = task
        self.agents = self.initialize_agents(agents_config, task)
        print(f"LLM_Debate initialized for task: {task} with {len(self.agents)} agents.")
    
    def initialize_agents(self, agents_config, task_type):
        agents = []
        devices = ['cuda:3', 'cuda:4', 'cuda:5']
        
        for i, config in enumerate(agents_config):
            device_id = devices[i % len(devices)]
            print(f"Initializing {config['agent_name']} on {device_id}")
            
            # print(task_type, config['agent_name'])
            agent_class = {"llama": LlamaAgent, "mistral": MistralAgent, "openai": OpenAIAgent}.get(config['type'])
            if not agent_class:
                raise ValueError(f"Unsupported agent type: {config['type']}")

            common_args = {
                "model_name": config['model_name'],
                "agent_name": config['agent_name'],
                "agent_role": config['agent_role'],
                "agent_speciality": config['agent_speciality'],
                "speaking_rate": config['speaking_rate'],
                "agent_type": config['type'],
                "device_id": device_id,  # Pass assigned device
            }

            if task_type == "SciFi-Review":
                agent = agent_class(
                    **common_args, 
                    agent_role_prompt=config['agent_role_prompt'], 
                    general_instruction=config.get('general_instruction')
                )
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            agents.append(agent)
        return agents

    
    def format_response(self, agent, response):
        return response
        # if agent.agent_type == "mistral":
        #     try:
        #         formatted_response = response.split("assistant")[-1]
        #     except:
        #         formatted_response = "The response has invalid format, ignore and proceed."
        #     # formatted_response = response
        # elif agent.agent_type == "llama":
        #     if "Teacher" in agent.agent_name:
        #         try:
        #             formatted_response = response.split("Your response must strictly adhere to this format.\n")[-1]
        #         except:
        #             try:
        #                 formatted_response = "Here's my critique:" + response.split("Here's my critique:")[-1]
        #             except:
        #                 formatted_response = "The response has invalid format, ignore and proceed."
        #     else:
        #         try:
        #             formatted_response = response.split("Your response must strictly adhere to this format.\n")[-1]
        #         except:
        #             try:
        #                 formatted_response = "Here's my story:" + response.split("Here's my story:")[-1]
        #             except:
        #                 formatted_response = "The response has invalid format, ignore and proceed."
        # return formatted_response

    def save_debate_conversations(self, agents, all_responses, init_results, final_results, amount_of_data, 
                                  final_logits=None, task_type="AUT", baseline=False):
        current_date, formatted_time = self.get_current_datetime()
        model_names_concatenated = self.concatenate_model_names(agents)
        role_names_concatenated = self.concatenate_role_names(agents)
        subtask = self.determine_subtask(agents, baseline)

        output_filename = self.generate_filename(task_type, subtask, "chat_log", model_names_concatenated, role_names_concatenated, current_date, formatted_time, amount_of_data, len(agents), self.rounds)
        final_ans_filename = self.generate_final_filename(task_type, subtask, "multi_agent", model_names_concatenated, role_names_concatenated, current_date, formatted_time, amount_of_data, len(agents), self.rounds)
        init_ans_filename = self.generate_filename(task_type, subtask, "init", model_names_concatenated, role_names_concatenated, current_date, formatted_time, amount_of_data, len(agents), self.rounds)

        # Save logits separately as .npy files to local storage
        if final_logits is not None:
            for logit_entry in final_logits:
                agent_name = logit_entry["agent_name"]
                logits = logit_entry["logit"]

                if logits is not None:
                    # Generate a filename for each agent's logits
                    logit_filename = self.generate_filename(
                        task_type, subtask, f"logits_{agent_name}", model_names_concatenated, role_names_concatenated,
                        current_date, formatted_time, amount_of_data, len(agents), self.rounds
                    )
                    logit_filename = logit_filename.replace(".json", ".npy")  # Save logits as a .npy file
                    
                    # Move logits to CPU and convert to NumPy before saving
                    if hasattr(logits, 'detach'):
                         logits_np = np.array(logits.detach().cpu().numpy())
                    else:
                         logits_np = np.array(logits)

                    
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(logit_filename), exist_ok=True)
                    
                    # Save to local file
                    np.save(logit_filename, logits_np)
                    print(f"Saved Logits to local file: {logit_filename}")

                    # Add the filename to the corresponding agent's entry in final_results
                    for result in final_results:
                        if result["Agent"] == agent_name:
                            result["logits_file"] = logit_filename  # Add the logit filename to the final_results entry

        # Save the conversations, including the logit filenames, to local storage
        self.save_conversation(output_filename, all_responses)
        self.save_conversation(final_ans_filename, final_results)
        self.save_conversation(init_ans_filename, init_results)

        return final_ans_filename
    
    @staticmethod
    def get_current_datetime():
        current_time = datetime.datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        formatted_time = current_time.strftime("%H-%M-%S")
        return current_date, formatted_time
    
    @staticmethod
    def concatenate_model_names(agents):
        if all(agent.model_name == agents[0].model_name for agent in agents):
            return agents[0].model_name.replace(".", "-")
        return "-".join(agent.model_name.replace(".", "-") for agent in agents)

    @staticmethod
    def concatenate_role_names(agents):
        if all(agent.agent_role == "None" for agent in agents):
            return "None"
        return "-".join(agent.agent_role.replace(" ", "") for agent in agents)

    def determine_subtask(self, agents, baseline):
        if baseline:
            return "baseline"
        if all(agent.agent_role == "None" for agent in agents):
            return "FINAL"
        return "roleplay"
    
    @staticmethod
    def generate_filename(task_type, subtask, data_type, model_names_concatenated, role_names_concatenated, current_date, formatted_time, amount_of_data, num_agents, num_rounds):
        return f"../../Results/{task_type}/{data_type}/{task_type}_multi_teacher_{subtask}_{num_agents}_{num_rounds}_{model_names_concatenated}_{role_names_concatenated}_{data_type}_{current_date}-{formatted_time}_{amount_of_data}.json"

    @staticmethod
    def generate_final_filename(task_type, subtask, data_type, model_names_concatenated, role_names_concatenated, current_date, formatted_time, amount_of_data, num_agents, num_rounds):
        return f"../../Results/{task_type}/Output/{data_type}/{task_type}_multi_teacher_{subtask}_{num_agents}_{num_rounds}_{model_names_concatenated}_{role_names_concatenated}_{data_type}_{current_date}-{formatted_time}_{amount_of_data}.json"

class LLM_Review_SciFi(LLM_Debate):
    def run(self):
        with open(self.dataset_file, 'r') as f:
            dataset = json.load(f)
        all_responses = {}
        init_results = []
        final_results = []
        final_logits = []
        all_examples = [d for key, val in dataset.items() for d in val]
        amount_of_data = 0
        
        # normal runs
        for example in all_examples:
            amount_of_data += 1      
            self.process_example(example, amount_of_data)
        
        # decoding experiment section
        # temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
        # top_ps = [0.7, 0.8, 0.9, 1.0]
        # top_ks = [5,10,50,100]
        # rep_penalties = [1.0, 1.2, 1.5]
        
        # all_sampled_groups = []
        # for rep_pen in rep_penalties:
        #     sampled_examples = []
        #     for key, val in dataset.items():
        #         # Sample 2 examples from each 'val' with replacement
        #         sampled_examples.extend(random.sample(val, k=2))

        #     # Ensure we have 20 examples in total for each group
        #     assert len(sampled_examples) == 20
        #     all_sampled_groups.append((sampled_examples, rep_pen))
        
        # for examples, rep_pen in all_sampled_groups:
        #     for example in examples:
        #         amount_of_data += 1      
        #         self.process_example(example, amount_of_data, rep_pen=rep_pen)
        
    def process_example(self, example, amount_of_data, temp=0.6, top_p=0.9, top_k=50, rep_pen=1.0):
        
        print('Processing prompt:', amount_of_data)
                
        all_responses = {}
        init_results = []
        final_results = []
        final_logits = []
        
        chat_history = {agent.agent_role: [] for agent in self.agents}
        question = example

        agent_stories = {agent.agent_role: [] for agent in self.agents}
        agent_reviews = {
            agent.agent_role: {
                other_agent.agent_role: []
                for other_agent in self.agents
                if other_agent.agent_role != agent.agent_role
            }
            for agent in self.agents
        }

        for round in range(self.rounds):
            is_last_round = (round == self.rounds - 1)
            is_first_round = (round == 0)
            print(f"Round {round + 1}: Discussion on '{question}'")

            # each agent generates a story first
            for agent in self.agents:
                if agent.agent_role != "None":
                    agent_role_prompt = (
                        f"You are a {agent.agent_role} whose specialty is "
                        f"{agent.agent_speciality}. Here's a breakdown of your role: "
                        f"{agent.agent_role_prompt} Besides that specialty, you should "
                        f"also follow some general instructions: {agent.general_instruction} "
                        "Remember to claim your role at the beginning of each conversation.\n"
                    )
                else:
                    agent_role_prompt = ""

                if is_first_round:
                    combined_prompt = self.construct_story_prompt_first_round(question, agent)
                else:
                    combined_prompt = self.construct_story_prompt(
                        question, agent, agent_stories, agent_reviews
                    )

                formatted_prompt = agent.construct_user_message(agent_role_prompt + combined_prompt)
                chat_history[agent.agent_role].append(formatted_prompt)

                # print(f"\n----- Agent {agent.agent_role} Prompt -----\n")
                # print(formatted_prompt)
                # print("\n------------------------------------------\n")

                if is_last_round:
                    raw_response, logits, entropy_max, entropy_mean, attention_entropy, surprisal_variance, surprisal_max, hidden_score = agent.generate_answer([formatted_prompt], final_round=True)
                    response = self.format_response(agent, raw_response)
                    final_result = {
                        "question": question,
                        "answer": [response],
                        "Agent": agent.agent_name,
                        "entropy_max": entropy_max.item() if isinstance(entropy_max, torch.Tensor) else entropy_max,
                        "entropy_mean": entropy_mean.item() if isinstance(entropy_mean, torch.Tensor) else entropy_mean,
                        "attention_entropy": attention_entropy,
                        "surprisal_variance": surprisal_variance,
                        "surprisal_max": surprisal_max,
                        "hidden_score": hidden_score
                    }
                    final_results.append(final_result)
                    final_logit = {"agent_name": agent.agent_name, "logit": logits}
                    final_logits.append(final_logit)
                else:
                    raw_response = agent.generate_answer([formatted_prompt])
                    response = self.format_response(agent, raw_response)
                    if is_first_round:
                        # save initial result
                        init_result = {
                            "question": question,
                            "answer": [response],
                            "Agent": agent.agent_name,
                        }
                        init_results.append(init_result)

                formatted_response = agent.construct_assistant_message(response)
                chat_history[agent.agent_role].append(formatted_response)
                agent_stories[agent.agent_role].append(response)

                # print(f"\n----- Agent {agent.agent_role} Response -----\n")
                # print(response)
                # print("\n---------------------------------------------\n")

            # Then, each agent reviews other agents' stories
            for agent in self.agents:
                if agent.agent_role != "None":
                    agent_role_prompt = (
                        f"You are a {agent.agent_role} whose specialty is "
                        f"{agent.agent_speciality}. Here's a breakdown of your role: "
                        f"{agent.agent_role_prompt} Besides that specialty, you should "
                        f"also follow some general instructions: {agent.general_instruction} "
                        "Remember to claim your role at the beginning of each conversation.\n"
                    )
                else:
                    agent_role_prompt = ""

                combined_prompt = self.construct_review_prompt(question, agent, agent_stories)
                formatted_prompt = agent.construct_user_message(agent_role_prompt + combined_prompt)
                chat_history[agent.agent_role].append(formatted_prompt)

                # print(f"\n----- Agent {agent.agent_role} Review Prompt -----\n")
                # print(formatted_prompt)
                # print("\n------------------------------------------------\n")

                raw_response = agent.generate_answer([formatted_prompt])
                response = self.format_response(agent, raw_response)

                formatted_response = agent.construct_assistant_message(response)
                chat_history[agent.agent_role].append(formatted_response)

                # print(f"\n----- Agent {agent.agent_role} Review Response -----\n")
                # print(response)
                # print("\n----------------------------------------------------\n")

                # parse and store the reviews
                reviews = self.parse_reviews(agent, response)
                # print(f"\n----- Extract Agent {agent.agent_role} Reviews -----\n")
                # print(reviews)
                # print("\n----------------------------------------------------\n")
                # print('----------agent role------------')
                # print(agent.agent_role)
                # print(reviews.keys())
                # print(agent_reviews)
                for other_agent_name, review in reviews.items():
                    if other_agent_name in agent_reviews:
                        if agent.agent_role in agent_reviews[other_agent_name]:
                            agent_reviews[other_agent_name][agent.agent_role].append(review)

        all_responses[question] = chat_history

        # save the conversations and results
        self.save_debate_conversations(
            self.agents,
            all_responses,
            init_results,
            final_results,
            amount_of_data,
            final_logits,
            task_type=self.task_type,
            )

    def construct_story_prompt_first_round(self, question, agent):
        format_string = f"""
        Please write a science fiction story based on the following prompt:
        {question}

        Please follow these formatting rules for your response:

        1. **Do not repeat the prompt**: Directly start generating the requested story without restating or summarizing the given prompt.
        2. **Start with the phrase**: "Here's my story:".
        3. **End with the phrase**: "End of Story".
        4. **Only include the story text between these phrases**: Avoid adding any commentary, explanation, or notes before or after the story.

        Do not ask questions or wait for guidance. Your response must strictly adhere to this format.\n
        """
        return format_string

    def construct_story_prompt(self, question, agent, agent_stories, agent_reviews):
        previous_story = agent_stories[agent.agent_role][-1]  # Get the agent's previous story
        reviews = [
            f"Feedback from {reviewer}:\n{agent_reviews[agent.agent_role][reviewer][-1]}"
            for reviewer in agent_reviews[agent.agent_role]
            if agent_reviews[agent.agent_role][reviewer]
        ]

        review_text = "\n".join(reviews) if reviews else "No feedback received yet."

        format_string = f"""
        Based on your previous story and the feedback from your peers, please improve your science fiction story.

        Your previous story:
        {previous_story}

        Feedback from your peers:
        {review_text}

        Please write a new version of your story, addressing the feedback. Follow the same formatting rules:

        1. **Do not repeat the prompt**: Directly start generating the requested story without restating or summarizing the given prompt.
        2. **Start with the phrase**: "Here's my story:".
        3. **End with the phrase**: "End of Story".
        4. **Only include the story text between these phrases**: Avoid adding any commentary, explanation, or notes before or after the story.

        Do not ask questions or wait for guidance. Your response must strictly adhere to this format.\n
        """
        return format_string

    def construct_review_prompt(self, question, agent, agent_stories):
        other_agents = [other_agent for other_agent in self.agents if other_agent.agent_role != agent.agent_name]
        other_stories = [
            (other_agent.agent_role, agent_stories[other_agent.agent_role][-1]) for other_agent in other_agents
        ]

        stories_text = "\n".join([f"{name}'s story:\n{story}" for name, story in other_stories])

        format_string = f"""
        Please review the following stories from your peers and provide constructive feedback for each one.

        {stories_text}

        Please format your feedback as follows:

        For each story, start with "Feedback for [Agent Name]:" on a new line, then provide your feedback.

        Focus your feedback on emphasizing creativity by encouraging the use of rare vocabulary, vivid imagery, and surprising plot elements. Highlight opportunities where the story could benefit from more imaginative settings, unique word choices, and unexpected twists. Ensure feedback also covers plot coherence, character development, and areas for improvement.

        Do not include any unrelated commentary or repeat the prompt. Your response must strictly adhere to this format.\n
        """
        return format_string

    def parse_reviews(self, agent, response):
        # Updated pattern to capture "Feedback for LLM Agent - xxx Writer's story"
        reviews = {}
        pattern = r"Feedback for ([\w\s\-]+):\s*(.+?)(?=(?:\n\s*)+Feedback for [\w\s\-]+:|$)"
        matches = re.findall(pattern, response, re.DOTALL)
        
        for agent_role, feedback in matches:
            reviews[agent_role.strip()] = feedback.strip()
        
        return reviews
