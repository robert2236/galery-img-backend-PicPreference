# model_metrics.py - VERSIÓN CORREGIDA
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
import io
import base64
from typing import List, Dict, Any
from database.databases import coleccion
from bson import ObjectId
from collections import Counter
import asyncio

metrics_router = APIRouter(prefix="/api/metrics", tags=["Model Metrics"])

# ============ FUNCIÓN AUXILIAR GLOBAL ============

def get_performance_rating(score: float) -> str:
    """Convierte score numérico a rating - FUNCIÓN GLOBAL"""
    if score >= 90:
        return "Excelente"
    elif score >= 80:
        return "Muy Bueno"
    elif score >= 70:
        return "Bueno"
    elif score >= 60:
        return "Aceptable"
    else:
        return "Necesita Mejora"

def get_interpretation(score: float) -> str:
    """Interpretación del score general"""
    if score >= 90:
        return "Rendimiento excepcional, sistema funcionando de manera óptima"
    elif score >= 80:
        return "Buen rendimiento, sistema estable y confiable"
    elif score >= 70:
        return "Rendimiento aceptable, algunas áreas podrían mejorar"
    elif score >= 60:
        return "Rendimiento básico, se recomiendan mejoras"
    else:
        return "Rendimiento insuficiente, requiere atención inmediata"

def generate_recommendations(summary: Dict) -> str:
    """Genera recomendaciones basadas en métricas"""
    recommendations = []
    score = summary.get('overall_performance', {}).get('score', 0)
    
    if score < 60:
        recommendations.append("⚠️ **ALERTA:** El rendimiento general es bajo. Revisar configuración de modelos.")
    elif score < 80:
        recommendations.append("ℹ️ **INFO:** Rendimiento aceptable. Considerar optimizaciones.")
    else:
        recommendations.append("✅ **EXCELENTE:** El sistema funciona óptimamente.")
    
    # Recomendaciones específicas
    if 'classification' in summary:
        f1 = summary['classification'].get('metrics', {}).get('avg_f1_score', 0)
        if f1 < 0.5:
            recommendations.append("🎯 **Clasificación:** Entrenar con más datos etiquetados para mejorar F1 Score.")
    
    if 'sentiment' in summary:
        corr = summary['sentiment'].get('correlation_metrics', {}).get('sentiment_likes_correlation', 0)
        if abs(corr) < 0.3:
            recommendations.append("💬 **Sentimiento:** El modelo no correlaciona bien con engagement. Revisar etiquetado.")
    
    if 'object_detection' in summary:
        avg_objects = summary['object_detection'].get('object_statistics', {}).get('avg_objects_per_image', 0)
        if avg_objects < 1:
            recommendations.append("🔍 **Detección:** El modelo detecta pocos objetos. Ajustar umbrales de confianza.")
    
    return "\n".join(f"- {rec}" for rec in recommendations)

def table_to_markdown(data: Dict) -> str:
    """Convierte diccionario a tabla markdown"""
    if not data:
        return ""
    
    table = "\n| Modelo | Métrica | Valor |\n|--------|---------|-------|\n"
    
    for category, metrics in data.items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                table += f"| {category} | {metric_name} | {value} |\n"
    
    return table

# ============ CLASE EVALUADORA ============

class ModelPerformanceEvaluator:
    """Evaluador de rendimiento de modelos IA"""
    
    @staticmethod
    async def evaluate_image_classification():
        """Evalúa el modelo de clasificación de imágenes"""
        try:
            # Obtener datos de etiquetado automático
            pipeline = [
                {
                    "$match": {
                        "ai_features.auto_tags": {"$exists": True, "$ne": []}
                    }
                },
                {
                    "$project": {
                        "ai_tags": "$ai_features.auto_tags",
                        "user_tags": "$tags",
                        "scene_type": "$ai_features.scene_type",
                        "title": 1
                    }
                },
                {"$limit": 500}
            ]
            
            
            data = await coleccion.aggregate(pipeline).to_list(length=None)
            
            if not data:
                return {"error": "No hay datos suficientes para evaluación"}
            
            # Preparar datos para evaluación
            evaluations = []
            
            for item in data:
                ai_tags = set(tag.lower() for tag in item.get('ai_tags', []))
                user_tags = set(tag.lower() for tag in item.get('user_tags', []))
                
                if user_tags:  # Solo evaluar si hay etiquetas de usuario
                    # Métricas básicas
                    common_tags = ai_tags.intersection(user_tags)
                    precision = len(common_tags) / len(ai_tags) if ai_tags else 0
                    recall = len(common_tags) / len(user_tags) if user_tags else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    evaluations.append({
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "common_tags": len(common_tags),
                        "total_ai_tags": len(ai_tags),
                        "total_user_tags": len(user_tags)
                    })
            
            if not evaluations:
                return {"error": "No hay coincidencias para evaluar"}
            
            df = pd.DataFrame(evaluations)
            
            # Métricas agregadas
            metrics = {
                "avg_precision": float(df['precision'].mean()),
                "avg_recall": float(df['recall'].mean()),
                "avg_f1_score": float(df['f1_score'].mean()),
                "total_evaluations": len(df),
                "avg_common_tags": float(df['common_tags'].mean()),
                "precision_std": float(df['precision'].std()),
                "recall_std": float(df['recall'].std())
            }
            
            # Gráficos de rendimiento
            charts = ModelPerformanceEvaluator._create_performance_charts(df)
            
            # Distribución de F1 scores
            fig_dist = px.histogram(
                df,
                x='f1_score',
                nbins=20,
                title="📊 Distribución de F1 Scores",
                color_discrete_sequence=['#FF9F43']
            )
            fig_dist.update_layout(
                xaxis_title="F1 Score",
                yaxis_title="Frecuencia",
                xaxis_range=[0, 1]
            )
            
            # Matriz de correlación
            correlation = df[['precision', 'recall', 'f1_score', 'common_tags']].corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig_corr.update_layout(title="🔥 Matriz de Correlación")
            
            charts["f1_distribution"] = fig_dist.to_html(full_html=False, include_plotlyjs='cdn')
            charts["correlation_matrix"] = fig_corr.to_html(full_html=False, include_plotlyjs='cdn')
            
            return {
                "metrics": metrics,
                "charts": charts,
                "sample_evaluations": evaluations[:10]
            }
            
        except Exception as e:
            return {"error": f"Error en evaluación de clasificación: {str(e)}"}
    
    @staticmethod
    async def evaluate_sentiment_analysis():
        """Evalúa el modelo de análisis de sentimiento"""
        try:
            # Obtener datos de sentimiento
            pipeline = [
                {
                    "$match": {
                        "social_features.sentiment_score": {"$exists": True},
                        "comments": {"$exists": True, "$ne": []}
                    }
                },
                {
                    "$project": {
                        "sentiment_score": "$social_features.sentiment_score",
                        "comments_count": {"$size": "$comments"},
                        "likes": "$interactions.likes",
                        "title": 1
                    }
                },
                {"$limit": 300}
            ]
            
            data = await coleccion.aggregate(pipeline).to_list(length=None)
            
            if not data:
                return {"error": "No hay datos de sentimiento para evaluación"}
            
            df = pd.DataFrame(data)
            
            # Análisis de correlación
            correlation_metrics = {}
            if 'likes' in df.columns and 'sentiment_score' in df.columns:
                corr = df['sentiment_score'].corr(df['likes'])
                correlation_metrics['sentiment_likes_correlation'] = float(corr) if not np.isnan(corr) else 0
            
            if 'comments_count' in df.columns and 'sentiment_score' in df.columns:
                corr = df['sentiment_score'].corr(df['comments_count'])
                correlation_metrics['sentiment_comments_correlation'] = float(corr) if not np.isnan(corr) else 0
            
            # Distribución de sentimientos
            sentiment_dist = {
                "very_positive": int((df['sentiment_score'] > 0.7).sum()),
                "positive": int(((df['sentiment_score'] > 0.3) & (df['sentiment_score'] <= 0.7)).sum()),
                "neutral": int(((df['sentiment_score'] >= -0.3) & (df['sentiment_score'] <= 0.3)).sum()),
                "negative": int(((df['sentiment_score'] < -0.3) & (df['sentiment_score'] >= -0.7)).sum()),
                "very_negative": int((df['sentiment_score'] < -0.7).sum())
            }
            
            # Gráficos
            charts = {}
            
            # Distribución de sentimiento
            fig1 = px.pie(
                values=list(sentiment_dist.values()),
                names=list(sentiment_dist.keys()),
                title="📊 Distribución de Sentimientos",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            charts["sentiment_distribution"] = fig1.to_html(full_html=False, include_plotlyjs='cdn')
            
            # Serie temporal de sentimiento (si hay datos de fecha)
            time_pipeline = [
                {
                    "$match": {
                        "social_features.sentiment_score": {"$exists": True},
                        "upload_date": {"$exists": True}
                    }
                },
                {
                    "$project": {
                        "sentiment_score": "$social_features.sentiment_score",
                        "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$upload_date"}}
                    }
                },
                {
                    "$group": {
                        "_id": "$date",
                        "avg_sentiment": {"$avg": "$sentiment_score"},
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"_id": 1}},
                {"$limit": 30}
            ]
            
            time_data = await coleccion.aggregate(time_pipeline).to_list(length=None)
            if time_data:
                df_time = pd.DataFrame(time_data)
                fig2 = px.line(
                    df_time,
                    x='_id',
                    y='avg_sentiment',
                    title="📈 Evolución del Sentimiento Promedio",
                    markers=True
                )
                fig2.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="Sentimiento Promedio",
                    yaxis_range=[-1, 1]
                )
                charts["sentiment_timeline"] = fig2.to_html(full_html=False, include_plotlyjs='cdn')
            
            return {
                "correlation_metrics": correlation_metrics,
                "sentiment_distribution": sentiment_dist,
                "summary_stats": {
                    "mean_sentiment": float(df['sentiment_score'].mean()) if len(df) > 0 else 0,
                    "std_sentiment": float(df['sentiment_score'].std()) if len(df) > 0 else 0,
                    "min_sentiment": float(df['sentiment_score'].min()) if len(df) > 0 else 0,
                    "max_sentiment": float(df['sentiment_score'].max()) if len(df) > 0 else 0
                },
                "charts": charts
            }
            
        except Exception as e:
            return {"error": f"Error en evaluación de sentimiento: {str(e)}"}
    
    @staticmethod
    async def evaluate_object_detection():
        """Evalúa el modelo de detección de objetos"""
        try:
            pipeline = [
                {
                    "$match": {
                        "ai_features.detected_objects": {"$exists": True, "$ne": []}
                    }
                },
                {
                    "$project": {
                        "detected_objects": "$ai_features.detected_objects",
                        "object_count": {"$size": "$ai_features.detected_objects"},
                        "title": 1,
                        "category": 1
                    }
                },
                {"$limit": 200}
            ]
            
            data = await coleccion.aggregate(pipeline).to_list(length=None)
            
            if not data:
                return {"error": "No hay datos de detección de objetos"}
            
            # Análisis de frecuencia de objetos
            all_objects = []
            for item in data:
                objects = item.get('detected_objects', [])
                if isinstance(objects, list):
                    all_objects.extend([str(obj).lower() for obj in objects if obj])
            
            object_freq = Counter(all_objects)
            
            # Top objetos detectados
            top_objects = dict(object_freq.most_common(15))
            
            # Distribución por categoría
            df = pd.DataFrame(data)
            category_analysis = {}
            if 'category' in df.columns:
                for category in df['category'].dropna().unique():
                    cat_objects = []
                    for idx, row in df[df['category'] == category].iterrows():
                        objects = row.get('detected_objects', [])
                        if isinstance(objects, list):
                            cat_objects.extend([str(obj).lower() for obj in objects if obj])
                    if cat_objects:
                        category_analysis[str(category)] = dict(Counter(cat_objects).most_common(5))
            
            # Gráficos
            charts = {}
            
            # Top objetos detectados
            if top_objects:
                fig1 = px.bar(
                    x=list(top_objects.keys()),
                    y=list(top_objects.values()),
                    title="🔍 Top 15 Objetos Detectados",
                    color=list(top_objects.values()),
                    color_continuous_scale='Viridis'
                )
                fig1.update_layout(
                    xaxis_title="Objeto",
                    yaxis_title="Frecuencia",
                    xaxis_tickangle=45
                )
                charts["top_objects"] = fig1.to_html(full_html=False, include_plotlyjs='cdn')
            
            # Distribución de conteo de objetos por imagen
            if 'object_count' in df.columns and len(df) > 0:
                fig2 = px.histogram(
                    df,
                    x='object_count',
                    nbins=15,
                    title="📊 Distribución de Objetos por Imagen",
                    color_discrete_sequence=['#00D2D3']
                )
                fig2.update_layout(
                    xaxis_title="Número de Objetos",
                    yaxis_title="Número de Imágenes"
                )
                charts["objects_per_image"] = fig2.to_html(full_html=False, include_plotlyjs='cdn')
            
            # Word cloud data (para visualización externa)
            wordcloud_data = [{"text": obj, "value": freq} for obj, freq in object_freq.most_common(50)]
            
            return {
                "object_statistics": {
                    "total_unique_objects": len(object_freq),
                    "total_detections": sum(object_freq.values()),
                    "avg_objects_per_image": float(df['object_count'].mean()) if 'object_count' in df.columns and len(df) > 0 else 0,
                    "top_objects": top_objects
                },
                "category_analysis": category_analysis,
                "wordcloud_data": wordcloud_data,
                "charts": charts
            }
            
        except Exception as e:
            return {"error": f"Error en evaluación de detección: {str(e)}"}
    
    @staticmethod
    async def get_model_benchmarks():
        """Obtiene benchmarks de rendimiento de modelos"""
        benchmarks = {
            "image_classification": {
                "model": "ResNet50",
                "expected_accuracy": 0.76,
                "expected_f1": 0.75,
                "inference_time_ms": 120,
                "memory_usage_mb": 256
            },
            "object_detection": {
                "model": "DETR-ResNet50",
                "mAP": 0.42,
                "inference_time_ms": 250,
                "memory_usage_mb": 512
            },
            "sentiment_analysis": {
                "model": "BERT-Multilingual",
                "expected_accuracy": 0.85,
                "expected_f1": 0.83,
                "inference_time_ms": 50
            }
        }
        
        return benchmarks
    
    @staticmethod
    def _create_performance_charts(df: pd.DataFrame) -> Dict:
        """Crea gráficos de rendimiento a partir de DataFrame"""
        charts = {}
        
        try:
            # Box plot de métricas
            fig_box = go.Figure()
            
            for metric in ['precision', 'recall', 'f1_score']:
                if metric in df.columns:
                    fig_box.add_trace(go.Box(
                        y=df[metric],
                        name=metric.capitalize(),
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8
                    ))
            
            fig_box.update_layout(
                title="📦 Distribución de Métricas de Rendimiento",
                yaxis_title="Score",
                yaxis_range=[0, 1]
            )
            charts["metrics_boxplot"] = fig_box.to_html(full_html=False, include_plotlyjs='cdn')
            
            # Scatter plot: Precision vs Recall
            if 'precision' in df.columns and 'recall' in df.columns:
                fig_scatter = px.scatter(
                    df,
                    x='precision',
                    y='recall',
                    size='common_tags',
                    color='f1_score',
                    hover_name=df.index if 'title' not in df.columns else df.get('title', df.index),
                    title="🎯 Precision vs Recall",
                    color_continuous_scale='RdYlGn'
                )
                fig_scatter.update_layout(
                    xaxis_title="Precision",
                    yaxis_title="Recall",
                    xaxis_range=[0, 1],
                    yaxis_range=[0, 1]
                )
                charts["precision_recall_scatter"] = fig_scatter.to_html(full_html=False, include_plotlyjs='cdn')
            
        except Exception as e:
            print(f"Error creando gráficos: {str(e)}")
        
        return charts

# ============ ENDPOINTS DE MÉTRICAS ============

@metrics_router.get("/classification")
async def get_classification_metrics():
    """Métricas de rendimiento de clasificación de imágenes"""
    try:
        return await ModelPerformanceEvaluator.evaluate_image_classification()
    except Exception as e:
        raise HTTPException(500, f"Error en clasificación: {str(e)}")

@metrics_router.get("/sentiment")
async def get_sentiment_metrics():
    """Métricas de rendimiento de análisis de sentimiento"""
    try:
        return await ModelPerformanceEvaluator.evaluate_sentiment_analysis()
    except Exception as e:
        raise HTTPException(500, f"Error en sentimiento: {str(e)}")

@metrics_router.get("/object-detection")
async def get_object_detection_metrics():
    """Métricas de rendimiento de detección de objetos"""
    try:
        return await ModelPerformanceEvaluator.evaluate_object_detection()
    except Exception as e:
        raise HTTPException(500, f"Error en detección: {str(e)}")

@metrics_router.get("/benchmarks")
async def get_model_benchmarks():
    """Benchmarks de todos los modelos"""
    return await ModelPerformanceEvaluator.get_model_benchmarks()

@metrics_router.get("/summary")
async def get_performance_summary():
    """Resumen completo del rendimiento de todos los modelos"""
    try:
        # Obtener datos de cada evaluación
        classification = await ModelPerformanceEvaluator.evaluate_image_classification()
        sentiment = await ModelPerformanceEvaluator.evaluate_sentiment_analysis()
        objects = await ModelPerformanceEvaluator.evaluate_object_detection()
        benchmarks = await ModelPerformanceEvaluator.get_model_benchmarks()
        
        # Calcular scores generales
        overall_score = 0
        component_scores = []
        errors = []
        
        # 1. Clasificación
        classification_score = 0
        classification_details = {}
        
        if 'error' not in classification and 'metrics' in classification:
            classification_score = classification['metrics'].get('avg_f1_score', 0) * 100
            classification_details = {
                "f1_score": classification['metrics'].get('avg_f1_score', 0),
                "precision": classification['metrics'].get('avg_precision', 0),
                "recall": classification['metrics'].get('avg_recall', 0),
                "evaluations": classification['metrics'].get('total_evaluations', 0)
            }
        else:
            errors.append(f"Clasificación: {classification.get('error', 'Error desconocido')}")
        
        component_scores.append({
            "component": "Clasificación de Imágenes",
            "score": round(classification_score, 2),
            "weight": 0.4,
            "status": "success" if 'error' not in classification else "error",
            "details": classification_details if classification_details else None
        })
        overall_score += classification_score * 0.4
        
        # 2. Sentimiento
        sentiment_score = 0
        sentiment_details = {}
        
        if 'error' not in sentiment and 'summary_stats' in sentiment:
            mean_sentiment = sentiment['summary_stats'].get('mean_sentiment', 0)
            sentiment_score = ((mean_sentiment + 1) / 2) * 100
            sentiment_details = {
                "mean_sentiment": mean_sentiment,
                "correlation_likes": sentiment.get('correlation_metrics', {}).get('sentiment_likes_correlation', 0),
                "distribution": sentiment.get('sentiment_distribution', {})
            }
        else:
            errors.append(f"Sentimiento: {sentiment.get('error', 'Error desconocido')}")
        
        component_scores.append({
            "component": "Análisis de Sentimiento",
            "score": round(sentiment_score, 2),
            "weight": 0.3,
            "status": "success" if 'error' not in sentiment else "error",
            "details": sentiment_details if sentiment_details else None
        })
        overall_score += sentiment_score * 0.3
        
        # 3. Detección de objetos
        object_score = 0
        object_details = {}
        
        if 'error' not in objects and 'object_statistics' in objects:
            avg_objects = objects['object_statistics'].get('avg_objects_per_image', 0)
            object_score = min((avg_objects / 3) * 100, 100) if avg_objects > 0 else 0
            object_details = {
                "avg_objects_per_image": avg_objects,
                "total_unique_objects": objects['object_statistics'].get('total_unique_objects', 0),
                "total_detections": objects['object_statistics'].get('total_detections', 0),
                "top_objects": list(objects['object_statistics'].get('top_objects', {}).items())[:3]
            }
        else:
            errors.append(f"Detección: {objects.get('error', 'Error desconocido')}")
        
        component_scores.append({
            "component": "Detección de Objetos",
            "score": round(object_score, 2),
            "weight": 0.3,
            "status": "success" if 'error' not in objects else "error",
            "details": object_details if object_details else None
        })
        overall_score += object_score * 0.3
        
        # Limitar score a 100
        overall_score = min(round(overall_score, 2), 100)
        
        # Usar la función global get_performance_rating CORREGIDA
        rating = get_performance_rating(overall_score)
        interpretation = get_interpretation(overall_score)
        
        # Preparar respuesta
        response = {
            "overall_performance": {
                "score": overall_score,
                "rating": rating,
                "interpretation": interpretation,
                "timestamp": datetime.utcnow().isoformat()
            },
            "component_scores": component_scores,
            "detailed_metrics": {
                "classification": classification,
                "sentiment": sentiment,
                "object_detection": objects
            },
            "benchmarks": benchmarks,
            "recommendations": generate_recommendations({
                "overall_performance": {"score": overall_score},
                "classification": classification,
                "sentiment": sentiment,
                "object_detection": objects
            })
        }
        
        # Agregar errores si existen
        if errors:
            response["errors"] = errors
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en resumen de rendimiento: {str(e)}")

@metrics_router.get("/export/report")
async def export_performance_report():
    """Exporta reporte completo de rendimiento"""
    try:
        summary = await get_performance_summary()
        
        # Crear reporte en formato markdown
        report = f"""
# 📊 Reporte de Rendimiento - Sistema IA
**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Puntuación general:** {summary['overall_performance']['score']}/100 ({summary['overall_performance']['rating']})
**Interpretación:** {summary['overall_performance']['interpretation']}

## 🔧 Componentes del Sistema

### 1. Clasificación de Imágenes
"""
        
        classification = summary['detailed_metrics']['classification']
        if 'error' not in classification and 'metrics' in classification:
            report += f"""
- **Precisión promedio:** {classification['metrics'].get('avg_precision', 0):.2%}
- **Recall promedio:** {classification['metrics'].get('avg_recall', 0):.2%}
- **F1 Score:** {classification['metrics'].get('avg_f1_score', 0):.2%}
- **Evaluaciones realizadas:** {classification['metrics'].get('total_evaluations', 0)}
"""
        else:
            report += f"- **Error:** {classification.get('error', 'No disponible')}\n"
        
        report += """
### 2. Análisis de Sentimiento
"""
        
        sentiment = summary['detailed_metrics']['sentiment']
        if 'error' not in sentiment and 'summary_stats' in sentiment:
            report += f"""
- **Sentimiento promedio:** {sentiment['summary_stats'].get('mean_sentiment', 0):.2f}
- **Desviación estándar:** {sentiment['summary_stats'].get('std_sentiment', 0):.2f}
- **Correlación con likes:** {sentiment.get('correlation_metrics', {}).get('sentiment_likes_correlation', 0):.2f}
"""
        else:
            report += f"- **Error:** {sentiment.get('error', 'No disponible')}\n"
        
        report += """
### 3. Detección de Objetos
"""
        
        detection = summary['detailed_metrics']['object_detection']
        if 'error' not in detection and 'object_statistics' in detection:
            report += f"""
- **Objetos únicos detectados:** {detection['object_statistics'].get('total_unique_objects', 0)}
- **Detecciones totales:** {detection['object_statistics'].get('total_detections', 0)}
- **Objetos por imagen:** {detection['object_statistics'].get('avg_objects_per_image', 0):.1f}
- **Top objeto:** {list(detection['object_statistics'].get('top_objects', {}).keys())[0] if detection['object_statistics'].get('top_objects') else 'N/A'}
"""
        else:
            report += f"- **Error:** {detection.get('error', 'No disponible')}\n"
        
        report += f"""
## 🎯 Benchmarks de Modelos
{table_to_markdown(summary['benchmarks'])}

## 📈 Recomendaciones
{summary['recommendations']}
"""
        
        # Agregar errores si existen
        if 'errors' in summary and summary['errors']:
            report += """
## ⚠️ Errores Encontrados

"""
            for error in summary['errors']:
                report += f"- {error}\n"
        
        report += f"""
---
*Reporte generado automáticamente por el Sistema de IA*
"""
        
        return {
            "report": report,
            "format": "markdown",
            "filename": f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "overall_score": summary['overall_performance']['score'],
                "rating": summary['overall_performance']['rating']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando reporte: {str(e)}")

@metrics_router.get("/health")
async def health_check():
    """Verifica salud del sistema de métricas"""
    try:
        # Verificar conexión a base de datos
        count = await coleccion.count_documents({})
        
        # Verificar disponibilidad de datos
        has_classification = await coleccion.count_documents({"ai_features.auto_tags": {"$exists": True}}) > 0
        has_sentiment = await coleccion.count_documents({"social_features.sentiment_score": {"$exists": True}}) > 0
        has_objects = await coleccion.count_documents({"ai_features.detected_objects": {"$exists": True}}) > 0
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": {
                "connected": True,
                "total_images": count
            },
            "data_availability": {
                "classification": has_classification,
                "sentiment": has_sentiment,
                "object_detection": has_objects
            },
            "endpoints": {
                "classification": "/api/metrics/classification",
                "sentiment": "/api/metrics/sentiment",
                "object_detection": "/api/metrics/object-detection",
                "summary": "/api/metrics/summary",
                "benchmarks": "/api/metrics/benchmarks",
                "export": "/api/metrics/export/report"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@metrics_router.get("/test")
async def test_endpoint():
    """Endpoint de prueba"""
    return {
        "status": "success",
        "message": "Sistema de métricas funcionando correctamente",
        "timestamp": datetime.utcnow().isoformat()
    }