import json
import os
import logging
import subprocess
import time
import numpy as np
from openai import OpenAI
import google.generativeai as genai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch.nn.functional as F

def calculate_attention_entropy(attentions):
    """
    Calculate average attention entropy across all tokens and heads.
    Uses float32 precision to avoid numerical instability.
    """
    entropy_values = []
    
    # attentions is a tuple, one per generated token
    for token_attentions in attentions:
        # last_layer: (batch_size, num_heads, seq_len, seq_len)
        last_layer = token_attentions[-1]
        
        # Average across batch (should be 1) and get per-head attention distributions
        # Cast to float32 for stability
        attention_weights = last_layer[0].float()  # (num_heads, seq_len, seq_len)
        
        # For each head, compute entropy of attention distribution for the last token
        for head_idx in range(attention_weights.shape[0]):
            # Get attention distribution for this head's last token
            attn_dist = attention_weights[head_idx, -1, :]  # (seq_len,)
            
            # Add small epsilon to avoid log(0)
            attn_dist = attn_dist + 1e-10
            attn_dist = attn_dist / attn_dist.sum()  # Normalize
            
            # Compute entropy: -Σ(p * log(p))
            entropy = -torch.sum(attn_dist * torch.log(attn_dist))
            if not torch.isnan(entropy):
                entropy_values.append(entropy.item())
    
    # Return average entropy across all tokens and heads
    return np.mean(entropy_values) if entropy_values else 0.0

def calculate_hidden_score(hidden_states):
    """
    Calculate Hidden Score using SVD of final-layer hidden states.
    Uses float32 precision to avoid numerical instability.
    """
    final_layer_states = []
    
    # Extract final layer hidden states for all generated tokens
    for token_hidden_states in hidden_states:
        # Take last layer: shape (batch_size, seq_len, hidden_dim)
        last_layer = token_hidden_states[-1]
        # Get hidden state for the last position in sequence
        # Cast to float32 for stability
        final_layer_states.append(last_layer[0, -1, :].float())
    
    # Stack to form matrix H: (num_tokens, hidden_dim)
    H = torch.stack(final_layer_states, dim=0)
    
    # Compute SVD: H = U @ diag(S) @ V^T
    try:
        # Determine effective rank or just compute SVD
        if H.shape[0] < 2:  # Need at least 2 tokens for meaningful covariance/SVD
            return 0.0
            
        U, S, V = torch.svd(H)
        
        # Check for NaNs in singular values
        if torch.isnan(S).any():
            return None
            
        # Hidden Score = (1/d) * Σ log(σᵢ²) = (1/d) * Σ 2*log(σᵢ)
        log_eigenvalues = 2 * torch.log(S + 1e-10)
        hidden_score = log_eigenvalues.mean().item()
        
        if np.isnan(hidden_score):
            return None
            
        return hidden_score
    except Exception as e:
        print(f"Error checking hidden score: {e}")
        return None

class Agent:
    def generate_answer(self, answer_context):
        raise NotImplementedError("This method should be implemented by subclasses.")
    def construct_assistant_message(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")
    def construct_user_message(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")

    
def find_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:1'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")
    return device
        
class LlamaAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, speaking_rate, agent_type,
                 agent_role_prompt=None, agent_role_prompt_initial=None, agent_role_prompt_followup=None, general_instruction=None, device_id=None):
        self.model_name = model_name
        self.agent_name = agent_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16,
            attn_implementation='eager'  # Required for output_attentions and output_hidden_states
        )
        
        if device_id:
            self.device = device_id
        else:
            self.device = find_device()
            if torch.cuda.device_count() > 1:
                # print(f"Using {torch.cuda.device_count()} GPUs")
                self.model = torch.nn.DataParallel(self.model)
        # self.device = 'cuda'
        self.model.to(self.device)
        self.agent_role = agent_role
        self.agent_type = agent_type
        self.agent_speciality = agent_speciality
        self.speaking_rate = speaking_rate
        self.agent_role_prompt = agent_role_prompt
        self.agent_role_prompt_initial = agent_role_prompt_initial
        self.agent_role_prompt_followup = agent_role_prompt_followup 
        self.general_instruction = general_instruction
        
    def generate_answer(self, answer_context, token_limit=1000, final_round=False, 
                        temp=0.6, top_p=0.9, top_k=50, rep_pen=1.0):
        inputs = self.tokenizer(answer_context[0]['content'], return_tensors="pt")

        # Move inputs to the same device as the model (MPS if available)
        inputs = {key: value.to(self.device) for key, value in inputs.items()} if self.tokenizer else inputs
        
        # Check if model is wrapped in DataParallel
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        # inputs = self.tokenizer(answer_context, return_tensors="pt")
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=token_limit,
            num_return_sequences=1,
            pad_token_id=model.config.eos_token_id[0],
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
            output_attentions=True,  # Enable attention capture
            output_hidden_states=True,  # Enable hidden states capture
            # temperature=temp,
            # top_p=top_p,
            # top_k=top_k,
            # repetition_penalty=rep_pen,
        )
        
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[:, input_length:]
        response_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Save logits if it's the final round and the agent is a student writer
        if final_round and "Writer" in self.agent_role:
            if isinstance(outputs.logits, tuple):
                logits = torch.cat(outputs.logits, dim=0)  # Replace `dim` based on your needs
            else:
                logits = outputs.logits
            
            probabilities = F.softmax(torch.tensor(logits), dim=-1)  # Shape: (sequence_length, vocab_size)
            max_probs, _ = torch.max(probabilities, dim=-1)  # choose max_probs
            entropy_max = -torch.sum(max_probs * torch.log2(max_probs)).item()
            
            entropy_per_time_step = -torch.sum(probabilities * torch.log2(probabilities), dim=-1)  # (sequence_length)
            entropy_mean = entropy_per_time_step.mean().item()
            
            # Calculate surprisal metrics (narrative unpredictability)
            chosen_probs = probabilities[range(len(generated_ids[0])), generated_ids[0]]
            surprisals = -torch.log(chosen_probs)
            surprisal_variance = surprisals.var().item()
            surprisal_max = surprisals.max().item()
            
            # Calculate Hidden Score (representational dimensionality)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_score = calculate_hidden_score(outputs.hidden_states)
            else:
                hidden_score = None
            
            # Calculate attention entropy
            if hasattr(outputs, 'attentions') and outputs.attentions:
                attention_entropy = calculate_attention_entropy(outputs.attentions)
            else:
                attention_entropy = None
            
            return response_text, logits, entropy_max, entropy_mean, attention_entropy, surprisal_variance, surprisal_max, hidden_score
            
        return response_text
    
    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}
    

class MistralAgent(Agent):

    def __init__(self, model_name, agent_name, agent_role, agent_speciality, speaking_rate, agent_type,
                 agent_role_prompt=None, agent_role_prompt_initial=None, agent_role_prompt_followup=None, general_instruction=None, device_id=None):
        self.model_name = model_name
        self.agent_name = agent_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16,
            attn_implementation='eager'  # Required for output_attentions and output_hidden_states
        )
        # Set device
        if device_id:
            self.device = device_id
        else:
            self.device = 'cuda'  # Assuming you're using GPU
        self.model.to(self.device)
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.speaking_rate = speaking_rate
        self.agent_type = agent_type
        self.agent_role_prompt = agent_role_prompt
        self.agent_role_prompt_initial = agent_role_prompt_initial
        self.agent_role_prompt_followup = agent_role_prompt_followup
        self.general_instruction = general_instruction

    def generate_answer(self, answer_context, token_limit=1000, final_round=False):
        # Tokenize input
        # inputs = self.tokenizer(answer_context[0]['content'], return_tensors="pt")
        encoded = self.tokenizer.apply_chat_template(answer_context, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        inputs = encoded.to(self.device)
        self.model.to(self.device)
        
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Generate output using Mistral model
        outputs = self.model.generate(
            # inputs,
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=token_limit,
            num_return_sequences=1,
            top_p=0.9,
            temperature=0.6,
            pad_token_id=self.model.config.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
            output_attentions=True,  # Enable attention capture
            output_hidden_states=True,  # Enable hidden states capture
            do_sample=True
        )

        # Return the decoded answer
        generated_ids = outputs.sequences.tolist()
        # print(len(generated_ids[0]))
        full_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        split_response = full_response.split("assistant\n", -1)
        response_text = split_response[1].strip() if len(split_response) > 1 else full_response
        print(response_text)
        # print(len(response_text))
        
        # Save logits if it's the final round and the agent is a student writer
        if final_round and "Writer" in self.agent_role:
            if isinstance(outputs.logits, tuple):
                logits = torch.cat(outputs.logits, dim=0)  # Replace `dim` based on your needs
            else:
                logits = outputs.logits
                
            # compute entropy
            probabilities = F.softmax(torch.tensor(logits), dim=-1)  # Shape: (sequence_length, vocab_size)
            max_probs, _ = torch.max(probabilities, dim=-1)  # choose max_probs
            entropy_max = -torch.sum(max_probs * torch.log2(max_probs)).item()
            
            entropy_per_time_step = -torch.sum(probabilities * torch.log2(probabilities), dim=-1)  # (sequence_length)
            entropy_mean = entropy_per_time_step.mean().item()
            
            # Calculate surprisal metrics (narrative unpredictability)
            # Need to get the generated portion only (excluding input)
            input_length = inputs.shape[1]
            generated_portion = outputs.sequences[0, input_length:]
            chosen_probs = probabilities[range(len(generated_portion)), generated_portion]
            surprisals = -torch.log(chosen_probs)
            surprisal_variance = surprisals.var().item()
            surprisal_max = surprisals.max().item()
            
            # Calculate Hidden Score (representational dimensionality)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_score = calculate_hidden_score(outputs.hidden_states)
            else:
                hidden_score = None
            
            # Calculate attention entropy
            if hasattr(outputs, 'attentions') and outputs.attentions:
                attention_entropy = calculate_attention_entropy(outputs.attentions)
            else:
                attention_entropy = None
            
            return response_text, logits, entropy_max, entropy_mean, attention_entropy, surprisal_variance, surprisal_max, hidden_score

        return response_text

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}

    def construct_user_message(self, content):
        return {"role": "user", "content": content}
class OpenAIAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, speaking_rate, agent_type,
                 agent_role_prompt=None, agent_role_prompt_initial=None, agent_role_prompt_followup=None, general_instruction=None, device_id=None):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.speaking_rate = speaking_rate
        self.agent_type = agent_type
        self.agent_role_prompt = agent_role_prompt
        self.agent_role_prompt_initial = agent_role_prompt_initial
        self.agent_role_prompt_followup = agent_role_prompt_followup
        self.general_instruction = general_instruction

    def generate_answer(self, answer_context, token_limit=1000, final_round=False, temp=0.6, top_p=0.9, top_k=50, rep_pen=1.0):
        # Prepare messages
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in answer_context]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=token_limit,
                temperature=temp,
                top_p=top_p,
                logprobs=True if final_round and "Writer" in self.agent_role else False,
                top_logprobs=5 if final_round and "Writer" in self.agent_role else None
            )

            response_text = response.choices[0].message.content

            if final_round and "Writer" in self.agent_role:
                # Basic entropy estimation from top_logprobs if available
                # Note: This is an approximation as we don't have full vocab distribution
                entropy_max = None
                entropy_mean = None
                surprisal_variance = None
                surprisal_max = None
                
                if response.choices[0].logprobs:
                    content_logprobs = response.choices[0].logprobs.content
                    if content_logprobs:
                         # Calculate surprisal/entropy from available token logprobs
                        logprobs = [token.logprob for token in content_logprobs]
                        probs = [np.exp(lp) for lp in logprobs]
                        surprisals = [-lp for lp in logprobs]
                        
                        surprisal_max = np.max(surprisals)
                        surprisal_variance = np.var(surprisals)
                        
                        # Entropy mean (of chosen tokens) - this is NOT full entropy but a proxy
                        entropy_mean = -np.mean([p * np.log2(p) for p in probs]) # Very rough proxy

                # Hidden score and attention entropy are not supported by OpenAI API
                hidden_score = None
                attention_entropy = None
                logits = None 

                return response_text, logits, entropy_max, entropy_mean, attention_entropy, surprisal_variance, surprisal_max, hidden_score

            return response_text

        except Exception as e:
            print(f"Error generating response from OpenAI: {e}")
            if final_round and "Writer" in self.agent_role:
                 return "", None, None, None, None, None, None, None
            return ""

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}

    def construct_user_message(self, content):
        return {"role": "user", "content": content}
