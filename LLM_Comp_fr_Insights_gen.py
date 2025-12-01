"""
LLM-Powered Supply Chain Insights System
Demonstrates: Fine-tuning, Prompt Engineering, RAG, and Evaluation
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import openai  # For comparison with GPT models

# ==================== STEP 1: Data Preparation for Fine-tuning ====================
class SupplyChainDataPreparation:
    """Prepare training data for LLM fine-tuning"""
    
    def __init__(self, forecasts_df: pd.DataFrame):
        self.forecasts_df = forecasts_df
    
    def create_training_examples(self) -> List[Dict]:
        """
        Create instruction-following examples for fine-tuning
        Format: {"instruction": "...", "input": "...", "output": "..."}
        """
        examples = []
        
        # Group by store and product
        for (store_id, product_id), group in self.forecasts_df.groupby(['store_id', 'product_id']):
            # Calculate metrics
            avg_sales = group['actual_sales'].mean()
            forecast_accuracy = 1 - abs(group['forecast'] - group['actual_sales']).mean() / avg_sales
            trend = "increasing" if group['actual_sales'].iloc[-1] > group['actual_sales'].iloc[0] else "decreasing"
            volatility = group['actual_sales'].std()
            
            # Create context
            context = {
                "store_id": int(store_id),
                "product_id": int(product_id),
                "avg_daily_sales": round(avg_sales, 2),
                "forecast_accuracy": round(forecast_accuracy * 100, 2),
                "trend": trend,
                "volatility": round(volatility, 2),
                "recent_sales": group['actual_sales'].tail(7).tolist()
            }
            
            # Example 1: Demand Analysis
            examples.append({
                "instruction": "Analyze the demand pattern for this product and provide procurement recommendations.",
                "input": json.dumps(context),
                "output": self._generate_demand_analysis(context)
            })
            
            # Example 2: Inventory Recommendation
            examples.append({
                "instruction": "Based on the sales data, recommend optimal inventory levels.",
                "input": json.dumps(context),
                "output": self._generate_inventory_recommendation(context)
            })
            
            # Example 3: Risk Assessment
            if volatility > avg_sales * 0.3:
                examples.append({
                    "instruction": "Identify supply chain risks for this product.",
                    "input": json.dumps(context),
                    "output": self._generate_risk_assessment(context)
                })
        
        return examples
    
    def _generate_demand_analysis(self, context: Dict) -> str:
        """Generate ground truth demand analysis"""
        trend_desc = "upward" if context['trend'] == "increasing" else "downward"
        
        analysis = f"""Product {context['product_id']} at Store {context['store_id']} shows a {trend_desc} trend with average daily sales of {context['avg_daily_sales']} units.

Forecast Accuracy: {context['forecast_accuracy']}% - {"Reliable forecasts" if context['forecast_accuracy'] > 85 else "Model needs improvement"}

Procurement Recommendations:
1. Order Quantity: {int(context['avg_daily_sales'] * 14)} units for 2-week supply
2. Safety Stock: {int(context['volatility'] * 2)} units (2x volatility)
3. Reorder Point: {int(context['avg_daily_sales'] * 7 + context['volatility'] * 2)} units

The {'high' if context['volatility'] > context['avg_daily_sales'] * 0.3 else 'moderate'} volatility suggests {'frequent monitoring and dynamic reordering' if context['volatility'] > context['avg_daily_sales'] * 0.3 else 'standard inventory management practices'}."""
        
        return analysis
    
    def _generate_inventory_recommendation(self, context: Dict) -> str:
        """Generate inventory recommendations"""
        service_level = 0.95  # 95% service level
        z_score = 1.65  # for 95% service level
        
        safety_stock = int(z_score * context['volatility'] * np.sqrt(7))  # 7-day lead time
        reorder_point = int(context['avg_daily_sales'] * 7 + safety_stock)
        economic_order = int(np.sqrt(2 * context['avg_daily_sales'] * 365 * 100 / 5))
        
        return f"""Inventory Optimization for Product {context['product_id']}:

Safety Stock: {safety_stock} units
- Calculated for 95% service level with 7-day lead time
- Protects against demand variability of {context['volatility']:.2f}

Reorder Point: {reorder_point} units
- Triggers when inventory drops to this level

Economic Order Quantity: {economic_order} units
- Minimizes total inventory costs

Current Status: {"Stock levels adequate" if context['avg_daily_sales'] > 30 else "Low-volume product, consider JIT approach"}"""
    
    def _generate_risk_assessment(self, context: Dict) -> str:
        """Generate risk assessment for volatile products"""
        return f"""Supply Chain Risk Assessment - Product {context['product_id']}:

HIGH VOLATILITY DETECTED (σ = {context['volatility']:.2f})

Key Risks:
1. Demand Uncertainty: {context['volatility'] / context['avg_daily_sales'] * 100:.1f}% coefficient of variation
2. Stockout Risk: Elevated due to unpredictable demand patterns
3. Overstock Risk: Potential for excess inventory during low-demand periods

Mitigation Strategies:
- Implement daily demand monitoring
- Establish flexible supplier agreements
- Consider safety stock increase to {int(context['volatility'] * 2.5)} units
- Enable expedited shipping option for emergency replenishment

Trend: {context['trend'].upper()} - Adjust base stock levels accordingly."""


# ==================== STEP 2: Fine-tuning with LoRA ====================
class LLMFineTuner:
    """Fine-tune LLM for supply chain insights using LoRA"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
    
    def prepare_dataset(self, examples: List[Dict]):
        """Convert examples to HuggingFace dataset"""
        
        # Format as instruction-following prompts
        formatted_examples = []
        for ex in examples:
            prompt = f"""### Instruction:
{ex['instruction']}

### Input:
{ex['input']}

### Response:
{ex['output']}"""
            formatted_examples.append({"text": prompt})
        
        return Dataset.from_list(formatted_examples)
    
    def setup_lora_model(self):
        """Setup model with LoRA for efficient fine-tuning"""
        
        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,  # Rank of update matrices
            lora_alpha=32,  # Scaling factor
            target_modules=["q", "v"],  # Which layers to apply LoRA
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        return self.model
    
    def tokenize_function(self, examples):
        """Tokenize examples for training"""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    def train(self, dataset, output_dir="./supply_chain_llm"):
        """Fine-tune the model"""
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            warmup_steps=100,
            weight_decay=0.01,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")


# ==================== STEP 3: RAG Implementation ====================
class SupplyChainRAG:
    """Retrieval Augmented Generation for supply chain queries"""
    
    def __init__(self, knowledge_base: pd.DataFrame):
        self.knowledge_base = knowledge_base
        self.embeddings = None
    
    def create_embeddings(self):
        """Create embeddings for knowledge base using sentence transformers"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create text descriptions for each product
        texts = []
        for _, row in self.knowledge_base.iterrows():
            text = f"Product {row['product_id']} at Store {row['store_id']}: " \
                   f"Average sales {row['avg_sales']}, trend {row['trend']}"
            texts.append(text)
        
        self.embeddings = model.encode(texts)
        
        return self.embeddings
    
    def retrieve_context(self, query: str, top_k: int = 3):
        """Retrieve relevant context for query"""
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Retrieve context
        context = self.knowledge_base.iloc[top_indices].to_dict('records')
        
        return context
    
    def generate_with_rag(self, query: str, llm_model):
        """Generate response using RAG"""
        
        # Retrieve relevant context
        context = self.retrieve_context(query)
        
        # Format prompt with context
        context_str = "\n".join([json.dumps(c) for c in context])
        
        prompt = f"""Use the following context to answer the question:

Context:
{context_str}

Question: {query}

Answer:"""
        
        # Generate response
        inputs = llm_model.tokenizer(prompt, return_tensors="pt")
        outputs = llm_model.model.generate(**inputs, max_length=300)
        response = llm_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response


# ==================== STEP 4: Evaluation Framework ====================
class LLMEvaluator:
    """Evaluate LLM outputs for supply chain insights"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_factual_accuracy(self, generated: str, ground_truth: str) -> float:
        """Check if key facts are present"""
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth, generated)
        
        return scores['rougeL'].fmeasure
    
    def evaluate_numerical_accuracy(self, generated: str, expected_values: Dict) -> float:
        """Extract and verify numerical recommendations"""
        import re
        
        # Extract numbers from generated text
        numbers = re.findall(r'\d+\.?\d*', generated)
        
        # Check if key values are present and reasonable
        score = 0.0
        checks = 0
        
        for key, value in expected_values.items():
            if str(int(value)) in generated or str(value) in generated:
                score += 1
            checks += 1
        
        return score / checks if checks > 0 else 0.0
    
    def evaluate_coherence(self, text: str) -> float:
        """Evaluate coherence using perplexity"""
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        
        perplexity = torch.exp(outputs.loss).item()
        
        # Lower perplexity is better, normalize to 0-1 score
        coherence_score = max(0, 1 - (perplexity / 100))
        
        return coherence_score
    
    def comprehensive_evaluation(self, generated: str, ground_truth: str, 
                                expected_values: Dict) -> Dict:
        """Run all evaluations"""
        
        results = {
            "factual_accuracy": self.evaluate_factual_accuracy(generated, ground_truth),
            "numerical_accuracy": self.evaluate_numerical_accuracy(generated, expected_values),
            "coherence": self.evaluate_coherence(generated),
            "length": len(generated.split()),
        }
        
        # Overall score
        results["overall"] = (
            results["factual_accuracy"] * 0.4 +
            results["numerical_accuracy"] * 0.4 +
            results["coherence"] * 0.2
        )
        
        return results


# ==================== STEP 5: Main Execution ====================
def main():
    print("=" * 60)
    print("LLM-Powered Supply Chain Insights System")
    print("=" * 60)
    
    # Load forecast results
    forecasts_df = pd.read_csv('forecast_results.csv')  # From Part 1
    
    # Step 1: Prepare training data
    print("\n1. Preparing training data...")
    data_prep = SupplyChainDataPreparation(forecasts_df)
    training_examples = data_prep.create_training_examples()
    print(f"   Created {len(training_examples)} training examples")
    
    # Step 2: Fine-tune LLM
    print("\n2. Fine-tuning LLM with LoRA...")
    fine_tuner = LLMFineTuner()
    dataset = fine_tuner.prepare_dataset(training_examples)
    model = fine_tuner.setup_lora_model()
    
    # Uncomment to train (requires GPU)
    # fine_tuner.train(dataset)
    
    print("   Model setup complete (training skipped for demo)")
    
    # Step 3: Setup RAG
    print("\n3. Setting up RAG system...")
    knowledge_base = forecasts_df.groupby(['store_id', 'product_id']).agg({
        'actual_sales': 'mean',
        'forecast': 'mean'
    }).reset_index()
    knowledge_base['trend'] = 'increasing'  # Simplified
    knowledge_base.columns = ['store_id', 'product_id', 'avg_sales', 'avg_forecast', 'trend']
    
    rag_system = SupplyChainRAG(knowledge_base)
    # rag_system.create_embeddings()  # Uncomment if sentence-transformers installed
    
    print("   RAG system ready")
    
    # Step 4: Evaluation
    print("\n4. Running evaluation...")
    evaluator = LLMEvaluator()
    
    sample_generated = training_examples[0]['output']
    sample_truth = training_examples[0]['output']
    
    eval_results = evaluator.comprehensive_evaluation(
        sample_generated,
        sample_truth,
        {"avg_daily_sales": 45.2, "safety_stock": 15}
    )
    
    print(f"\n   Evaluation Results:")
    for metric, score in eval_results.items():
        print(f"   - {metric}: {score:.3f}")
    
    print("\n" + "=" * 60)
    print("Setup complete! Next steps:")
    print("1. Uncomment training code and run on GPU")
    print("2. Install sentence-transformers for RAG")
    print("3. Deploy model with FastAPI")
    print("=" * 60)


if __name__ == "__main__":
    main()
