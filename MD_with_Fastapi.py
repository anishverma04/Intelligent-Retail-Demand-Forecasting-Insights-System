"""
Production Deployment for Supply Chain ML System
FastAPI + Docker + Model Serving
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
import tensorflow as tf
import numpy as np
import joblib
from datetime import datetime, timedelta
import uvicorn
import logging

# ==================== STEP 1: API Models ====================
class ForecastRequest(BaseModel):
    """Request model for demand forecasting"""
    store_id: int = Field(..., description="Store identifier")
    product_id: int = Field(..., description="Product identifier")
    historical_data: List[Dict] = Field(
        ..., 
        description="14 days of historical data",
        example=[{
            "date": "2024-11-01",
            "sales": 45,
            "price": 10.5,
            "promotion": 0,
            "inventory": 100,
            "day_of_week": 4,
            "month": 11,
            "is_holiday": 0
        }]
    )
    framework: str = Field("pytorch", description="Framework to use: pytorch or tensorflow")


class ForecastResponse(BaseModel):
    """Response model for forecasts"""
    store_id: int
    product_id: int
    forecast: List[float] = Field(..., description="7-day forecast")
    confidence_intervals: Optional[List[Dict]] = None
    insights: Optional[str] = None
    generated_at: datetime


class InsightRequest(BaseModel):
    """Request for LLM-generated insights"""
    store_id: int
    product_id: int
    context: Dict = Field(..., description="Sales and inventory context")
    query: str = Field(..., description="Specific question about supply chain")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: Dict[str, bool]
    timestamp: datetime


# ==================== STEP 2: Model Manager ====================
class ModelManager:
    """Manages loading and inference for multiple models"""
    
    def __init__(self):
        self.pytorch_model = None
        self.tensorflow_model = None
        self.llm_model = None
        self.scalers = {}
        self.logger = logging.getLogger(__name__)
    
    def load_pytorch_model(self, model_path: str = "demand_forecaster_pytorch.pth"):
        """Load PyTorch model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Recreate model architecture (import from training script)
            from pytorch_demand_forecast import LSTMDemandForecaster
            
            self.pytorch_model = LSTMDemandForecaster(
                input_size=7,
                hidden_size=128,
                num_layers=2,
                forecast_horizon=7
            )
            
            self.pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            self.pytorch_model.eval()
            
            self.scalers['pytorch_X'] = checkpoint['scaler_X']
            self.scalers['pytorch_y'] = checkpoint['scaler_y']
            
            self.logger.info("PyTorch model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model: {e}")
            return False
    
    def load_tensorflow_model(self, model_path: str = "demand_forecaster_full.h5"):
        """Load TensorFlow model"""
        try:
            self.tensorflow_model = tf.keras.models.load_model(model_path)
            
            self.scalers['tf_X'] = joblib.load('scaler_X.pkl')
            self.scalers['tf_y'] = joblib.load('scaler_y.pkl')
            
            self.logger.info("TensorFlow model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load TensorFlow model: {e}")
            return False
    
    def load_llm_model(self, model_path: str = "./supply_chain_llm"):
        """Load fine-tuned LLM"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.llm_model.eval()
            
            self.logger.info("LLM model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load LLM model: {e}")
            return False
    
    def predict_pytorch(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference with PyTorch model"""
        with torch.no_grad():
            # Scale input
            input_scaled = self.scalers['pytorch_X'].transform(
                input_data.reshape(-1, input_data.shape[-1])
            ).reshape(input_data.shape)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0)
            
            # Predict
            output = self.pytorch_model(input_tensor)
            
            # Denormalize
            output_denorm = self.scalers['pytorch_y'].inverse_transform(
                output.numpy()
            )
            
            return output_denorm[0]
    
    def predict_tensorflow(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference with TensorFlow model"""
        # Scale input
        input_scaled = self.scalers['tf_X'].transform(
            input_data.reshape(-1, input_data.shape[-1])
        ).reshape(input_data.shape)
        
        # Predict
        input_tensor = np.expand_dims(input_scaled, axis=0)
        output = self.tensorflow_model.predict(input_tensor, verbose=0)
        
        # Denormalize
        output_denorm = self.scalers['tf_y'].inverse_transform(output)
        
        return output_denorm[0]
    
    def generate_insights(self, prompt: str, max_length: int = 300) -> str:
        """Generate insights using LLM"""
        if self.llm_model is None:
            return "LLM model not loaded"
        
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


# ==================== STEP 3: FastAPI Application ====================
# Initialize FastAPI
app = FastAPI(
    title="Supply Chain ML API",
    description="Demand forecasting and insights generation API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== STEP 4: API Endpoints ====================
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Loading models...")
    
    # Load models (uncomment when models are available)
    # model_manager.load_pytorch_model()
    # model_manager.load_tensorflow_model()
    # model_manager.load_llm_model()
    
    logger.info("Startup complete")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Supply Chain ML API",
        "version": "1.0.0",
        "endpoints": ["/forecast", "/insights", "/health"]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "pytorch": model_manager.pytorch_model is not None,
            "tensorflow": model_manager.tensorflow_model is not None,
            "llm": model_manager.llm_model is not None
        },
        timestamp=datetime.now()
    )


@app.post("/forecast", response_model=ForecastResponse)
async def forecast_demand(request: ForecastRequest):
    """
    Generate demand forecast for a product
    """
    try:
        # Validate input data
        if len(request.historical_data) != 14:
            raise HTTPException(
                status_code=400,
                detail="Historical data must contain exactly 14 days"
            )
        
        # Convert to numpy array
        input_data = np.array([[
            item['sales'],
            item['price'],
            item['promotion'],
            item['inventory'],
            item['day_of_week'],
            item['month'],
            item['is_holiday']
        ] for item in request.historical_data])
        
        # Generate forecast
        if request.framework == "pytorch":
            if model_manager.pytorch_model is None:
                raise HTTPException(status_code=503, detail="PyTorch model not loaded")
            forecast = model_manager.predict_pytorch(input_data)
        else:
            if model_manager.tensorflow_model is None:
                raise HTTPException(status_code=503, detail="TensorFlow model not loaded")
            forecast = model_manager.predict_tensorflow(input_data)
        
        # Calculate confidence intervals (simple approach)
        std_dev = np.std([item['sales'] for item in request.historical_data])
        confidence_intervals = [
            {
                "day": i + 1,
                "lower": max(0, forecast[i] - 1.96 * std_dev),
                "upper": forecast[i] + 1.96 * std_dev
            }
            for i in range(len(forecast))
        ]
        
        return ForecastResponse(
            store_id=request.store_id,
            product_id=request.product_id,
            forecast=forecast.tolist(),
            confidence_intervals=confidence_intervals,
            insights=None,  # Can be populated with LLM
            generated_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insights")
async def generate_insights(request: InsightRequest):
    """
    Generate natural language insights using LLM
    """
    try:
        # Create prompt
        prompt = f"""### Instruction:
{request.query}

### Input:
Store: {request.store_id}
Product: {request.product_id}
Context: {request.context}

### Response:"""
        
        # Generate insights
        if model_manager.llm_model is None:
            # Fallback to rule-based insights
            insights = generate_rule_based_insights(request.context)
        else:
            insights = model_manager.generate_insights(prompt)
        
        return {
            "store_id": request.store_id,
            "product_id": request.product_id,
            "query": request.query,
            "insights": insights,
            "generated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-forecast")
async def batch_forecast(
    requests: List[ForecastRequest],
    background_tasks: BackgroundTasks
):
    """
    Process multiple forecasts in batch
    """
    results = []
    
    for req in requests:
        try:
            result = await forecast_demand(req)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch forecast error for {req.store_id}-{req.product_id}: {e}")
            results.append({
                "store_id": req.store_id,
                "product_id": req.product_id,
                "error": str(e)
            })
    
    return {"results": results, "total": len(requests), "successful": len([r for r in results if "error" not in r])}


# ==================== STEP 5: Helper Functions ====================
def generate_rule_based_insights(context: Dict) -> str:
    """Generate insights without LLM (fallback)"""
    avg_sales = context.get('avg_sales', 0)
    trend = context.get('trend', 'stable')
    volatility = context.get('volatility', 0)
    
    insights = f"""Demand Analysis:
- Average daily sales: {avg_sales:.2f} units
- Trend: {trend}
- Volatility: {'High' if volatility > avg_sales * 0.3 else 'Moderate'}

Recommendations:
- Safety stock: {int(volatility * 2)} units
- Reorder point: {int(avg_sales * 7 + volatility * 2)} units
- Review frequency: {'Daily' if volatility > avg_sales * 0.3 else 'Weekly'}
"""
    
    return insights


# ==================== STEP 6: Run Server ====================
if __name__ == "__main__":
    uvicorn.run(
        "deployment_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
