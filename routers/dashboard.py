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
    
# ============ MÉTODOS NUEVOS PARA GRÁFICAS ADICIONALES ============


    @staticmethod
    async def create_engagement_analysis():
        try:
            pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"interactions.likes": {"$exists": True}},
                            {"interactions.shares": {"$exists": True}},
                            {"comments": {"$exists": True}}  # Cambiado: ahora busca el array comments
                        ]
                    }
                },
                {
                    "$project": {
                        "category": {"$ifNull": ["$category", "Sin categoría"]},
                        "likes": {"$ifNull": ["$interactions.likes", 0]},
                        "comments": {"$size": {"$ifNull": ["$comments", []]}},  # Cambiado: cuenta el array de comentarios
                        "shares": {"$ifNull": ["$interactions.shares", 0]},
                        "views": {"$ifNull": ["$interactions.views", 0]},
                        "title": {"$ifNull": ["$title", "Sin título"]},
                        "upload_date": 1
                    }
                },
                {
                    "$match": {
                        "$or": [
                            {"likes": {"$gt": 0}},
                            {"comments": {"$gt": 0}},
                            {"shares": {"$gt": 0}}
                        ]
                    }
                },
                {"$limit": 500}
            ]
            
            data = await coleccion.aggregate(pipeline).to_list(length=None)
            
            if not data or len(data) == 0:
                return {
                    "success": False,
                    "error": "No hay datos de engagement",
                    "suggestion": "Asegúrate de que las imágenes tengan campos de interacciones"
                }
            
            df = pd.DataFrame(data)
            
            # ====== CONVERSIÓN SEGURA DE TIPOS DE DATOS ======
            numeric_columns = ['likes', 'comments', 'shares', 'views']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)
                    df[col] = df[col].astype(int)
            
            if 'category' in df.columns:
                df['category'] = df['category'].astype(str)
            
            # ====== DATOS PARA GRÁFICOS (JSON) ======
            chart_data = {}
            
            # 1. Gráfico de barras: Engagement por categoría
            if 'category' in df.columns and len(df['category'].unique()) > 1:
                engagement_by_category = []
                
                for category in df['category'].unique():
                    cat_df = df[df['category'] == category]
                    
                    if len(cat_df) > 0:
                        engagement_by_category.append({
                            'category': category,
                            'avg_likes': float(cat_df['likes'].mean()),
                            'avg_comments': float(cat_df['comments'].mean()),
                            'avg_shares': float(cat_df['shares'].mean()),
                            'image_count': len(cat_df)
                        })
                
                if engagement_by_category:
                    # Ordenar por likes (descendente) y limitar a top 10
                    engagement_by_category_sorted = sorted(
                        engagement_by_category, 
                        key=lambda x: x['avg_likes'], 
                        reverse=True
                    )[:10]
                    
                    chart_data["engagement_by_category"] = {
                        "type": "bar",
                        "title": "📊 Engagement Promedio por Categoría (Top 10)",
                        "xAxis": {
                            "title": "Categoría",
                            "categories": [item['category'] for item in engagement_by_category_sorted]
                        },
                        "yAxis": {
                            "title": "Engagement Promedio"
                        },
                        "series": [
                            {
                                "name": "Likes",
                                "data": [item['avg_likes'] for item in engagement_by_category_sorted],
                                "color": "#3498db"
                            },
                            {
                                "name": "Comentarios",
                                "data": [item['avg_comments'] for item in engagement_by_category_sorted],
                                "color": "#2ecc71"
                            },
                            {
                                "name": "Shares",
                                "data": [item['avg_shares'] for item in engagement_by_category_sorted],
                                "color": "#9b59b6"
                            }
                        ]
                    }
            
            #2 Scatter data
            scatter_data = df[(df['likes'] > 0) & (df['comments'] > 0)].copy()
            
            print(f"📊 Datos para scatter plot: {len(scatter_data)} imágenes con likes y comentarios")
            
            if len(scatter_data) >= 3:  # Reducido a 3 para que sea más flexible
                if 'category' not in scatter_data.columns:
                    scatter_data['category'] = 'General'
                
                # Preparar datos para Highcharts scatter
                scatter_points_data = []
                
                for idx, (_, row) in enumerate(scatter_data.iterrows()):
                    scatter_points_data.append([
                        int(row['likes']),
                        int(row['comments']),
                        str(row['title'])[:30] if 'title' in row else f"Imagen {idx+1}",
                        str(row['category'])
                    ])
                
                chart_data["likes_vs_comments"] = {
                    "type": "scatter",
                    "title": "🎯 Relación: Likes vs Comentarios",
                    "subtitle": f"{len(scatter_data)} imágenes con ambos datos",
                    "xAxis": {
                        "title": "Likes",
                        "min": 0
                    },
                    "yAxis": {
                        "title": "Comentarios",
                        "min": 0
                    },
                    "series": [{
                        "name": "Imágenes",
                        "data": scatter_points_data,
                        "color": "rgba(52, 152, 219, 0.7)",
                        "marker": {
                            "radius": 6,
                            "symbol": "circle"
                        }
                    }],
                    "tooltip": {
                        "headerFormat": '<b>{point.point.name}</b><br/>',
                        "pointFormat": 'Likes: <b>{point.x}</b><br/>Comentarios: <b>{point.y}</b><br/>Categoría: {point.category}'
                    }
                }
            else:
                print(f"⚠️ No hay suficientes datos para scatter plot (necesita al menos 3 imágenes con likes y comentarios)")
                # Crear un scatter plot con datos de solo likes o solo comentarios
                scatter_alt_data = df[(df['likes'] > 0) | (df['comments'] > 0)].copy()
                
                if len(scatter_alt_data) >= 3:
                    scatter_points_data = []
                    
                    for idx, (_, row) in enumerate(scatter_alt_data.iterrows()):
                        scatter_points_data.append([
                            int(row['likes']),
                            int(row['comments']),
                            str(row['title'])[:30] if 'title' in row else f"Imagen {idx+1}",
                            str(row['category']) if 'category' in row else "General"
                        ])
                    
                    chart_data["likes_vs_comments"] = {
                        "type": "scatter",
                        "title": "🎯 Relación: Likes vs Comentarios",
                        "subtitle": f"{len(scatter_alt_data)} imágenes (algunas pueden tener solo likes o comentarios)",
                        "xAxis": {
                            "title": "Likes",
                            "min": 0
                        },
                        "yAxis": {
                            "title": "Comentarios",
                            "min": 0
                        },
                        "series": [{
                            "name": "Imágenes",
                            "data": scatter_points_data,
                            "color": "rgba(52, 152, 219, 0.7)",
                            "marker": {
                                "radius": 6
                            }
                        }]
                    }
            
            # 3. Top 10 imágenes con más engagement
            if len(df) >= 3:
                # Calcular engagement score
                df['engagement_score'] = (
                    df['likes'] * 0.5 + 
                    df['comments'] * 0.3 + 
                    df['shares'] * 0.2
                )
                
                # Top imágenes
                top_n = min(10, len(df))
                top_images = df.nlargest(top_n, 'engagement_score')
                
                top_images_data = []
                for idx, (_, row) in enumerate(top_images.iterrows()):
                    top_images_data.append({
                        'id': idx + 1,
                        'title': str(row['title']) if 'title' in row else f"Imagen {idx+1}",
                        'likes': int(row['likes']),
                        'comments': int(row['comments']),
                        'shares': int(row['shares']) if 'shares' in row else 0,
                        'engagement_score': float(row['engagement_score']),
                        'category': str(row['category']) if 'category' in row else "Sin categoría"
                    })
                
                chart_data["top_engagement_images"] = {
                    "type": "bar",
                    "title": f"🏆 Top {top_n} Imágenes con Más Engagement",
                    "xAxis": {
                        "title": "Imagen",
                        "categories": [item['title'] for item in top_images_data]
                    },
                    "yAxis": {
                        "title": "Interacciones"
                    },
                    "series": [
                        {
                            "name": "Likes",
                            "data": [item['likes'] for item in top_images_data],
                            "color": "#e74c3c"
                        },
                        {
                            "name": "Comentarios",
                            "data": [item['comments'] for item in top_images_data],
                            "color": "#f39c12"
                        },
                        {
                            "name": "Shares",
                            "data": [item['shares'] for item in top_images_data],
                            "color": "#3498db"
                        }
                    ],
                    "table_data": top_images_data  # Datos adicionales para tabla
                }
            
            # 4. Distribución de engagement (Box plot)
            metrics_for_boxplot = []
            for metric in ['likes', 'comments', 'shares']:
                if metric in df.columns and df[metric].sum() > 0:
                    metrics_for_boxplot.append(metric)
            
            if len(metrics_for_boxplot) >= 1:
                boxplot_data = {}
                for metric in metrics_for_boxplot:
                    metric_data = df[df[metric] > 0][metric].tolist()
                    if len(metric_data) > 0:
                        boxplot_data[metric] = {
                            "data": [float(x) for x in metric_data],
                            "color": '#3498db' if metric == 'likes' else 
                                    '#2ecc71' if metric == 'comments' else '#9b59b6'
                        }
                
                if boxplot_data:
                    chart_data["engagement_distribution"] = {
                        "type": "boxplot",
                        "title": "📦 Distribución de Engagement",
                        "yAxis": {
                            "title": "Cantidad"
                        },
                        "data": boxplot_data
                    }
            
            # 5. Gráfico de dona: Proporción de tipos de interacción
            total_interactions = {
                'Likes': int(df['likes'].sum()),
                'Comentarios': int(df['comments'].sum()),
                'Shares': int(df['shares'].sum())
            }
            
            # Filtrar solo los que tienen datos
            total_interactions_filtered = {k: v for k, v in total_interactions.items() if v > 0}
            
            if len(total_interactions_filtered) >= 2:
                pie_data = []
                colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
                
                for idx, (name, value) in enumerate(total_interactions_filtered.items()):
                    pie_data.append({
                        'name': name,
                        'value': value,
                        'color': colors[idx % len(colors)],
                        'percentage': round((value / sum(total_interactions_filtered.values())) * 100, 2)
                    })
                
                chart_data["interaction_proportion"] = {
                    "type": "pie",
                    "title": "🍩 Proporción de Tipos de Interacción",
                    "data": pie_data
                }
            
            # 6. Timeline de engagement (si hay fechas)
            if 'upload_date' in df.columns:
                try:
                    df['upload_date'] = pd.to_datetime(df['upload_date'])
                    df['date'] = df['upload_date'].dt.date
                    
                    # Agrupar por día
                    daily_stats = df.groupby('date').agg({
                        'likes': 'sum',
                        'comments': 'sum',
                        'shares': 'sum',
                        'title': 'count'
                    }).reset_index()
                    daily_stats.columns = ['date', 'total_likes', 'total_comments', 'total_shares', 'image_count']
                    
                    # Ordenar por fecha
                    daily_stats = daily_stats.sort_values('date')
                    
                    if len(daily_stats) >= 3:
                        timeline_dates = [row['date'].strftime('%Y-%m-%d') for _, row in daily_stats.iterrows()]
                        
                        chart_data["engagement_timeline"] = {
                            "type": "line",
                            "title": "📈 Evolución Temporal del Engagement",
                            "xAxis": {
                                "title": "Fecha",
                                "categories": timeline_dates
                            },
                            "yAxis": {
                                "title": "Interacciones"
                            },
                            "series": [
                                {
                                    "name": "Likes",
                                    "data": [int(row['total_likes']) for _, row in daily_stats.iterrows()],
                                    "color": "#3498db"
                                },
                                {
                                    "name": "Comentarios",
                                    "data": [int(row['total_comments']) for _, row in daily_stats.iterrows()],
                                    "color": "#2ecc71"
                                },
                                {
                                    "name": "Shares",
                                    "data": [int(row['total_shares']) for _, row in daily_stats.iterrows()],
                                    "color": "#9b59b6"
                                }
                            ]
                        }
                except Exception:
                    # Si hay error con fechas, simplemente omitir
                    pass
            
            # ====== ESTADÍSTICAS RESUMEN ======
            summary_stats = {
                "total_images": len(df),
                "images_with_engagement": int(((df['likes'] > 0) | (df['comments'] > 0) | (df['shares'] > 0)).sum()),
                "totals": {
                    "likes": int(df['likes'].sum()),
                    "comments": int(df['comments'].sum()),
                    "shares": int(df['shares'].sum()),
                    "views": int(df['views'].sum()) if 'views' in df.columns else 0
                },
                "averages": {
                    "likes_per_image": float(df['likes'].mean()),
                    "comments_per_image": float(df['comments'].mean()),
                    "shares_per_image": float(df['shares'].mean())
                },
                "maximums": {
                    "max_likes": int(df['likes'].max()) if len(df) > 0 else 0,
                    "max_comments": int(df['comments'].max()) if len(df) > 0 else 0,
                    "max_shares": int(df['shares'].max()) if len(df) > 0 else 0
                }
            }
            
            # Imagen más popular
            if len(df) > 0:
                max_likes_idx = df['likes'].idxmax()
                summary_stats["most_popular_image"] = {
                    "title": df.loc[max_likes_idx, 'title'] if 'title' in df.columns else "N/A",
                    "likes": int(df.loc[max_likes_idx, 'likes']),
                    "comments": int(df.loc[max_likes_idx, 'comments']),
                    "shares": int(df.loc[max_likes_idx, 'shares']),
                    "category": df.loc[max_likes_idx, 'category'] if 'category' in df.columns else "N/A"
                }
            
            # Categorías más populares
            if 'category' in df.columns and len(df['category'].unique()) > 1:
                category_popularity = df.groupby('category').agg({
                    'likes': 'sum',
                    'comments': 'sum',
                    'shares': 'sum'
                }).sum(axis=1).sort_values(ascending=False).head(3)
                
                summary_stats["top_categories"] = [
                    {
                        "category": cat,
                        "total_engagement": int(total)
                    }
                    for cat, total in category_popularity.items()
                ]
            
            return {
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "summary": summary_stats,
                "charts": chart_data,
                "metadata": {
                    "chart_count": len(chart_data),
                    "available_charts": list(chart_data.keys()),
                    "total_records": len(df),
                    "categories_count": len(df['category'].unique()) if 'category' in df.columns else 0
                }
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": f"Error en análisis de engagement: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    @staticmethod
    async def create_temporal_analysis():
        """Análisis temporal de subida de imágenes - VERSIÓN JSON PARA REACT"""
        try:
            pipeline = [
                {
                    "$match": {
                        "upload_date": {"$exists": True}
                    }
                },
                {
                    "$project": {
                        "upload_date": 1,
                        "category": {"$ifNull": ["$category", "Sin categoría"]},
                        "likes": {"$ifNull": ["$interactions.likes", 0]},
                        "comments": {"$ifNull": ["$interactions.comments", 0]},
                        "shares": {"$ifNull": ["$interactions.shares", 0]},
                        "title": {"$ifNull": ["$title", "Sin título"]}
                    }
                },
                {"$sort": {"upload_date": 1}}
            ]
            
            data = await coleccion.aggregate(pipeline).to_list(length=None)
            
            if not data or len(data) == 0:
                return {
                    "success": False,
                    "error": "No hay datos temporales",
                    "suggestion": "Asegúrate de que las imágenes tengan campo upload_date"
                }
            
            df = pd.DataFrame(data)
            
            # Convertir fechas de manera segura
            try:
                df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
                # Eliminar filas con fechas inválidas
                df = df.dropna(subset=['upload_date'])
                
                if len(df) == 0:
                    return {
                        "success": False,
                        "error": "No hay fechas válidas para análisis"
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error procesando fechas: {str(e)}"
                }
            
            # Convertir campos numéricos
            numeric_columns = ['likes', 'comments', 'shares']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)
                    df[col] = df[col].astype(int)
            
            # ====== DATOS PARA GRÁFICOS (JSON) ======
            chart_data = {}
            
            # 1. Línea temporal: Imágenes subidas por día
            df['date'] = df['upload_date'].dt.date
            daily_stats = df.groupby('date').agg({
                '_id': 'count',
                'likes': 'sum',
                'comments': 'sum',
                'shares': 'sum'
            }).reset_index()
            daily_stats.columns = ['date', 'image_count', 'total_likes', 'total_comments', 'total_shares']
            
            # Ordenar por fecha
            daily_stats = daily_stats.sort_values('date')
            
            if len(daily_stats) >= 2:
                # Datos para gráfico de líneas
                dates_str = [date.strftime('%Y-%m-%d') for date in daily_stats['date']]
                
                # Calcular promedio móvil (7 días)
                daily_stats['moving_avg'] = daily_stats['image_count'].rolling(
                    window=min(7, len(daily_stats)), 
                    min_periods=1
                ).mean()
                
                chart_data["upload_frequency"] = {
                    "type": "line",
                    "title": "📈 Frecuencia de Subida de Imágenes",
                    "xAxis": {
                        "title": "Fecha",
                        "categories": dates_str
                    },
                    "yAxis": {
                        "title": "Imágenes Subidas"
                    },
                    "series": [
                        {
                            "name": "Imágenes por día",
                            "data": daily_stats['image_count'].tolist(),
                            "color": "#3498db",
                            "type": "line+markers"
                        },
                        {
                            "name": "Promedio Móvil (7 días)",
                            "data": daily_stats['moving_avg'].tolist(),
                            "color": "#e74c3c",
                            "type": "line",
                            "dashed": True
                        }
                    ]
                }
            
            # 2. Heatmap por día de la semana y hora
            try:
                df['day_of_week'] = df['upload_date'].dt.day_name()
                df['hour'] = df['upload_date'].dt.hour
                
                # Traducir días al español
                days_translation = {
                    'Monday': 'Lunes',
                    'Tuesday': 'Martes', 
                    'Wednesday': 'Miércoles',
                    'Thursday': 'Jueves',
                    'Friday': 'Viernes',
                    'Saturday': 'Sábado',
                    'Sunday': 'Domingo'
                }
                df['day_of_week_es'] = df['day_of_week'].map(days_translation)
                
                # Agrupar para heatmap
                heatmap_raw = df.groupby(['day_of_week_es', 'hour']).size().reset_index(name='count')
                
                # Crear matriz para heatmap
                days_order_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                hours = list(range(24))
                
                # Inicializar matriz con ceros
                heatmap_matrix = []
                for day in days_order_es:
                    day_data = []
                    for hour in hours:
                        value = heatmap_raw[
                            (heatmap_raw['day_of_week_es'] == day) & 
                            (heatmap_raw['hour'] == hour)
                        ]['count'].sum()
                        day_data.append(int(value))
                    heatmap_matrix.append(day_data)
                
                if sum(sum(row) for row in heatmap_matrix) > 0:
                    chart_data["activity_heatmap"] = {
                        "type": "heatmap",
                        "title": "🔥 Actividad por Día y Hora",
                        "xAxis": {
                            "title": "Hora del Día",
                            "categories": [f"{h:02d}:00" for h in hours]
                        },
                        "yAxis": {
                            "title": "Día de la Semana",
                            "categories": days_order_es
                        },
                        "data": heatmap_matrix,
                        "colorscale": "Viridis"
                    }
            except Exception as e:
                # Si hay error en heatmap, continuar con otros gráficos
                print(f"Error en heatmap: {e}")
            
            # 3. Gráfico de área: Acumulado de imágenes
            if len(daily_stats) >= 2:
                daily_stats['cumulative'] = daily_stats['image_count'].cumsum()
                
                chart_data["cumulative_images"] = {
                    "type": "area",
                    "title": "📊 Total Acumulado de Imágenes",
                    "xAxis": {
                        "title": "Fecha",
                        "categories": dates_str
                    },
                    "yAxis": {
                        "title": "Imágenes Totales"
                    },
                    "series": [
                        {
                            "name": "Imágenes acumuladas",
                            "data": daily_stats['cumulative'].tolist(),
                            "color": "#27ae60",
                            "fill": True
                        }
                    ]
                }
            
            # 4. Gráfico de barras: Actividad por mes
            try:
                df['month'] = df['upload_date'].dt.to_period('M').astype(str)
                monthly_stats = df.groupby('month').agg({
                    '_id': 'count',
                    'likes': 'sum',
                    'comments': 'sum'
                }).reset_index()
                monthly_stats.columns = ['month', 'image_count', 'total_likes', 'total_comments']
                
                if len(monthly_stats) >= 2:
                    chart_data["monthly_activity"] = {
                        "type": "bar",
                        "title": "📅 Actividad por Mes",
                        "xAxis": {
                            "title": "Mes",
                            "categories": monthly_stats['month'].tolist()
                        },
                        "yAxis": {
                            "title": "Cantidad"
                        },
                        "series": [
                            {
                                "name": "Imágenes",
                                "data": monthly_stats['image_count'].tolist(),
                                "color": "#3498db"
                            },
                            {
                                "name": "Likes",
                                "data": monthly_stats['total_likes'].tolist(),
                                "color": "#e74c3c"
                            },
                            {
                                "name": "Comentarios",
                                "data": monthly_stats['total_comments'].tolist(),
                                "color": "#2ecc71"
                            }
                        ]
                    }
            except Exception as e:
                print(f"Error en estadísticas mensuales: {e}")
            
            # 5. Gráfico de dispersión: Engagement vs Fecha
            if len(df) >= 5:
                scatter_temporal = []
                for _, row in df.iterrows():
                    scatter_temporal.append({
                        'date': row['upload_date'].strftime('%Y-%m-%d'),
                        'likes': int(row['likes']),
                        'comments': int(row['comments']),
                        'shares': int(row['shares']),
                        'total_engagement': int(row['likes'] + row['comments'] + row['shares']),
                        'category': str(row['category']) if 'category' in row else "General"
                    })
                
                chart_data["engagement_over_time"] = {
                    "type": "scatter",
                    "title": "🎯 Engagement por Fecha",
                    "xAxis": {
                        "title": "Fecha",
                        "type": "date"
                    },
                    "yAxis": {
                        "title": "Engagement Total"
                    },
                    "data": scatter_temporal
                }
            
            # ====== ESTADÍSTICAS RESUMEN ======
            summary_stats = {
                "total_images": len(df),
                "date_range": {
                    "first_date": df['upload_date'].min().strftime('%Y-%m-%d') if len(df) > 0 else "N/A",
                    "last_date": df['upload_date'].max().strftime('%Y-%m-%d') if len(df) > 0 else "N/A",
                    "days_span": (df['upload_date'].max() - df['upload_date'].min()).days if len(df) > 1 else 0
                },
                "daily_stats": {
                    "avg_images_per_day": float(daily_stats['image_count'].mean()) if len(daily_stats) > 0 else 0,
                    "max_images_per_day": int(daily_stats['image_count'].max()) if len(daily_stats) > 0 else 0,
                    "total_days": len(daily_stats)
                }
            }
            
            # Calcular día y hora más activos si hay heatmap
            if "activity_heatmap" in chart_data:
                heatmap_matrix = chart_data["activity_heatmap"]["data"]
                days_order_es = chart_data["activity_heatmap"]["yAxis"]["categories"]
                
                # Encontrar día más activo
                day_sums = [sum(row) for row in heatmap_matrix]
                if sum(day_sums) > 0:
                    busiest_day_idx = day_sums.index(max(day_sums))
                    busiest_day = days_order_es[busiest_day_idx]
                    
                    # Encontrar hora más activa
                    hour_sums = [sum(col) for col in zip(*heatmap_matrix)]
                    if sum(hour_sums) > 0:
                        busiest_hour_idx = hour_sums.index(max(hour_sums))
                        busiest_hour = f"{busiest_hour_idx:02d}:00"
                        
                        summary_stats["activity_patterns"] = {
                            "busiest_day": busiest_day,
                            "busiest_hour": busiest_hour,
                            "peak_activity": int(max(hour_sums))
                        }
            
            # Estadísticas por categoría temporal
            if 'category' in df.columns and len(df['category'].unique()) > 1:
                category_trend = df.groupby(['date', 'category']).size().reset_index(name='count')
                category_trend_pivot = category_trend.pivot(index='date', columns='category', values='count').fillna(0)
                
                top_categories = df['category'].value_counts().head(3)
                summary_stats["top_categories_temporal"] = [
                    {"category": cat, "count": int(count)} 
                    for cat, count in top_categories.items()
                ]
            
            return {
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "summary": summary_stats,
                "charts": chart_data,
                "metadata": {
                    "chart_count": len(chart_data),
                    "available_charts": list(chart_data.keys()),
                    "date_range_days": summary_stats["date_range"]["days_span"],
                    "data_points": len(df)
                }
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": f"Error en análisis temporal: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }

    @staticmethod
    async def create_category_analysis():
        """Análisis detallado por categorías - VERSIÓN JSON PARA REACT"""
        try:
            pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"category": {"$exists": True, "$ne": None}},
                            {"ai_features.scene_type": {"$exists": True}}
                        ]
                    }
                },
                {
                    "$project": {
                        "category": {"$ifNull": ["$category", "$ai_features.scene_type", "Sin categoría"]},
                        "tags": {"$ifNull": ["$tags", []]},
                        "likes": {"$ifNull": ["$interactions.likes", 0]},
                        "comments": {"$ifNull": ["$interactions.comments", 0]},
                        "shares": {"$ifNull": ["$interactions.shares", 0]},
                        "sentiment_score": {"$ifNull": ["$social_features.sentiment_score", 0]},
                        "title": {"$ifNull": ["$title", "Sin título"]},
                        "detected_objects": {"$ifNull": ["$ai_features.detected_objects", []]}
                    }
                }
            ]
            
            data = await coleccion.aggregate(pipeline).to_list(length=None)
            
            if not data or len(data) == 0:
                return {
                    "success": False,
                    "error": "No hay datos por categoría",
                    "suggestion": "Asegúrate de que las imágenes tengan categorías asignadas"
                }
            
            df = pd.DataFrame(data)
            
            # Convertir campos numéricos de manera segura
            numeric_columns = ['likes', 'comments', 'shares', 'sentiment_score']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)
            
            # Asegurar que category sea string
            if 'category' in df.columns:
                df['category'] = df['category'].astype(str).str.strip()
                # Reemplazar categorías vacías
                df['category'] = df['category'].replace(['', 'None', 'nan'], 'Sin categoría')
            
            # ====== DATOS PARA GRÁFICOS (JSON) ======
            chart_data = {}
            
            # 1. Estadísticas por categoría
            category_stats_list = []
            
            for category in df['category'].unique():
                cat_df = df[df['category'] == category]
                
                if len(cat_df) > 0:
                    # Calcular estadísticas
                    stats = {
                        'category': category,
                        'image_count': len(cat_df),
                        'avg_likes': float(cat_df['likes'].mean()),
                        'avg_comments': float(cat_df['comments'].mean()),
                        'avg_shares': float(cat_df['shares'].mean()),
                        'avg_sentiment': float(cat_df['sentiment_score'].mean()),
                        'total_likes': int(cat_df['likes'].sum()),
                        'total_comments': int(cat_df['comments'].sum()),
                        'total_shares': int(cat_df['shares'].sum()),
                        'engagement_score': float(
                            (cat_df['likes'].mean() * 0.5) +
                            (cat_df['comments'].mean() * 0.3) +
                            (cat_df['shares'].mean() * 0.2)
                        )
                    }
                    
                    # Calcular objetos más comunes en esta categoría
                    if 'detected_objects' in cat_df.columns:
                        all_objects = []
                        for objects in cat_df['detected_objects']:
                            if isinstance(objects, list):
                                all_objects.extend([str(obj).lower() for obj in objects if obj])
                        
                        if all_objects:
                            from collections import Counter
                            object_counter = Counter(all_objects)
                            stats['top_objects'] = [
                                {'object': obj, 'count': count}
                                for obj, count in object_counter.most_common(5)
                            ]
                    
                    # Calcular etiquetas más comunes
                    if 'tags' in cat_df.columns:
                        all_tags = []
                        for tags in cat_df['tags']:
                            if isinstance(tags, list):
                                all_tags.extend([str(tag).lower() for tag in tags if tag])
                        
                        if all_tags:
                            from collections import Counter
                            tag_counter = Counter(all_tags)
                            stats['top_tags'] = [
                                {'tag': tag, 'count': count}
                                for tag, count in tag_counter.most_common(5)
                            ]
                    
                    category_stats_list.append(stats)
            
            if not category_stats_list:
                return {
                    "success": False,
                    "error": "No se pudieron calcular estadísticas por categoría"
                }
            
            # Ordenar por cantidad de imágenes (descendente)
            category_stats_list.sort(key=lambda x: x['image_count'], reverse=True)
            
            # 2. Datos para treemap de categorías
            top_categories = category_stats_list[:15]  # Limitar a top 15 para visualización
            
            chart_data["category_treemap"] = {
                "type": "treemap",
                "title": "🌳 Distribución de Categorías",
                "subtitle": "Tamaño = Cantidad de Imágenes, Color = Engagement",
                "data": [
                    {
                        "category": item['category'],
                        "value": item['image_count'],
                        "color_value": item['engagement_score'],
                        "details": {
                            "avg_likes": item['avg_likes'],
                            "avg_comments": item['avg_comments'],
                            "engagement_score": item['engagement_score']
                        }
                    }
                    for item in top_categories
                ],
                "color_scale": "RdYlGn"
            }
            
            # 3. Datos para gráfico de barras: Comparación de categorías
            if len(category_stats_list) >= 3:
                top_for_bar = category_stats_list[:8]  # Top 8 para gráfico de barras
                
                chart_data["category_comparison"] = {
                    "type": "bar",
                    "title": "📊 Comparación de Categorías (Top 8)",
                    "xAxis": {
                        "title": "Categoría",
                        "categories": [item['category'] for item in top_for_bar]
                    },
                    "yAxis": {
                        "title": "Valor Promedio"
                    },
                    "series": [
                        {
                            "name": "Likes Promedio",
                            "data": [item['avg_likes'] for item in top_for_bar],
                            "color": "#3498db"
                        },
                        {
                            "name": "Comentarios Promedio",
                            "data": [item['avg_comments'] for item in top_for_bar],
                            "color": "#2ecc71"
                        },
                        {
                            "name": "Sentimiento Promedio",
                            "data": [(item['avg_sentiment'] + 1) / 2 for item in top_for_bar],  # Normalizar a 0-1
                            "color": "#9b59b6"
                        }
                    ]
                }
            
            # 4. Datos para gráfico de burbujas: Cantidad vs Engagement
            if len(category_stats_list) >= 5:
                chart_data["category_bubbles"] = {
                    "type": "bubble",
                    "title": "🌀 Engagement vs Cantidad por Categoría",
                    "xAxis": {
                        "title": "Cantidad de Imágenes",
                        "min": 0
                    },
                    "yAxis": {
                        "title": "Likes Promedio",
                        "min": 0
                    },
                    "data": [
                        {
                            "category": item['category'],
                            "x": item['image_count'],
                            "y": item['avg_likes'],
                            "size": min(item['avg_comments'] * 10, 100),  # Escalar tamaño
                            "color": item['engagement_score'],
                            "details": {
                                "comments_avg": item['avg_comments'],
                                "shares_avg": item['avg_shares'],
                                "sentiment": item['avg_sentiment']
                            }
                        }
                        for item in category_stats_list[:20]  # Limitar a 20 categorías
                    ]
                }
            
            # 5. Datos para gráfico de radar (solo si hay suficientes categorías y métricas)
            if len(category_stats_list) >= 3:
                top_for_radar = category_stats_list[:5]  # Top 5 para radar
                
                # Normalizar valores para radar (0-1)
                max_values = {
                    'image_count': max(item['image_count'] for item in top_for_radar),
                    'avg_likes': max(item['avg_likes'] for item in top_for_radar),
                    'avg_comments': max(item['avg_comments'] for item in top_for_radar),
                    'avg_sentiment': max((item['avg_sentiment'] + 1) / 2 for item in top_for_radar)  # Normalizado
                }
                
                radar_series = []
                for item in top_for_radar:
                    radar_series.append({
                        "category": item['category'],
                        "values": [
                            item['image_count'] / max_values['image_count'] if max_values['image_count'] > 0 else 0,
                            item['avg_likes'] / max_values['avg_likes'] if max_values['avg_likes'] > 0 else 0,
                            item['avg_comments'] / max_values['avg_comments'] if max_values['avg_comments'] > 0 else 0,
                            ((item['avg_sentiment'] + 1) / 2) / max_values['avg_sentiment'] if max_values['avg_sentiment'] > 0 else 0
                        ],
                        "color": f"hsl({(len(radar_series) * 60) % 360}, 70%, 60%)"  # Colores distintos
                    })
                
                chart_data["category_radar"] = {
                    "type": "radar",
                    "title": "📡 Comparación Multidimensional de Categorías",
                    "dimensions": ["Cantidad", "Likes", "Comentarios", "Sentimiento"],
                    "series": radar_series
                }
            
            # 6. Datos para gráfico de torta: Distribución porcentual
            total_images = sum(item['image_count'] for item in category_stats_list)
            if total_images > 0:
                pie_data = []
                colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#34495e', '#d35400']
                
                for idx, item in enumerate(category_stats_list[:8]):  # Top 8 para torta
                    percentage = (item['image_count'] / total_images) * 100
                    pie_data.append({
                        "category": item['category'],
                        "value": item['image_count'],
                        "percentage": round(percentage, 2),
                        "color": colors[idx % len(colors)],
                        "avg_engagement": item['engagement_score']
                    })
                
                # Agrupar el resto como "Otras"
                other_count = sum(item['image_count'] for item in category_stats_list[8:])
                if other_count > 0:
                    other_percentage = (other_count / total_images) * 100
                    pie_data.append({
                        "category": "Otras categorías",
                        "value": other_count,
                        "percentage": round(other_percentage, 2),
                        "color": "#95a5a6",
                        "avg_engagement": 0
                    })
                
                chart_data["category_distribution"] = {
                    "type": "pie",
                    "title": "🥧 Distribución de Categorías",
                    "data": pie_data,
                    "total_images": total_images
                }
            
            # ====== ESTADÍSTICAS RESUMEN ======
            summary_stats = {
                "total_categories": len(category_stats_list),
                "total_images": total_images,
                "category_with_most_images": {
                    "category": category_stats_list[0]['category'] if category_stats_list else "N/A",
                    "image_count": category_stats_list[0]['image_count'] if category_stats_list else 0,
                    "percentage": round((category_stats_list[0]['image_count'] / total_images * 100), 2) if category_stats_list else 0
                },
                "category_with_best_engagement": {
                    "category": max(category_stats_list, key=lambda x: x['engagement_score'])['category'] if category_stats_list else "N/A",
                    "engagement_score": max(category_stats_list, key=lambda x: x['engagement_score'])['engagement_score'] if category_stats_list else 0
                },
                "category_with_best_sentiment": {
                    "category": max(category_stats_list, key=lambda x: x['avg_sentiment'])['category'] if category_stats_list else "N/A",
                    "sentiment_score": max(category_stats_list, key=lambda x: x['avg_sentiment'])['avg_sentiment'] if category_stats_list else 0
                }
            }
            
            # Estadísticas de categorías principales
            if len(category_stats_list) >= 3:
                top_3_categories = category_stats_list[:3]
                summary_stats["top_categories"] = [
                    {
                        "rank": i + 1,
                        "category": cat['category'],
                        "image_count": cat['image_count'],
                        "percentage": round((cat['image_count'] / total_images) * 100, 2),
                        "avg_likes": cat['avg_likes'],
                        "avg_comments": cat['avg_comments']
                    }
                    for i, cat in enumerate(top_3_categories)
                ]
            
            # Distribución de imágenes por categoría
            distribution_stats = []
            for item in category_stats_list:
                distribution_stats.append({
                    "category": item['category'],
                    "image_count": item['image_count'],
                    "percentage": round((item['image_count'] / total_images) * 100, 2),
                    "avg_engagement": item['engagement_score'],
                    "avg_sentiment": item['avg_sentiment']
                })
            
            return {
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "summary": summary_stats,
                "charts": chart_data,
                "detailed_stats": distribution_stats,
                "metadata": {
                    "chart_count": len(chart_data),
                    "available_charts": list(chart_data.keys()),
                    "categories_analyzed": len(category_stats_list),
                    "total_images_analyzed": total_images
                }
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {
                "success": False,
                "error": f"Error en análisis de categorías: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    @staticmethod
    async def create_tag_analysis():
        """Análisis de etiquetas y palabras clave - VERSIÓN CORREGIDA"""
        try:
            # Obtener la colección de MongoDB desde tu conexión existente
            # Reemplaza 'coleccion' con cómo obtienes tu colección actualmente
            # Ejemplo: coleccion = db["nombre_coleccion"]
            
            # NOTA: Asegúrate de que 'coleccion' esté disponible en el ámbito actual
            # Si usas una variable global, agrégala aquí
            from database import get_db  # Ajusta según tu estructura
            
            db = await get_db()
            coleccion = db["your_collection_name"]  # Reemplaza con tu nombre de colección
            
            pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"tags": {"$exists": True, "$ne": []}},
                            {"ai_features.auto_tags": {"$exists": True, "$ne": []}}
                        ]
                    }
                },
                {
                    "$project": {
                        "user_tags": "$tags",
                        "ai_tags": "$ai_features.auto_tags",
                        "category": 1
                    }
                },
                {"$limit": 300}
            ]
            
            data = await coleccion.aggregate(pipeline).to_list(length=None)
            
            if not data:
                return {
                    "success": False,
                    "error": "No hay datos de etiquetas",
                    "data": None
                }
            
            # Recolectar todas las etiquetas
            all_user_tags = []
            all_ai_tags = []
            
            for item in data:
                if isinstance(item.get('user_tags'), list):
                    all_user_tags.extend([tag.lower() for tag in item['user_tags'] if tag])
                if isinstance(item.get('ai_tags'), list):
                    all_ai_tags.extend([tag.lower() for tag in item['ai_tags'] if tag])
            
            # Conteo de frecuencia
            user_tag_freq = Counter(all_user_tags)
            ai_tag_freq = Counter(all_ai_tags)
            
            # Top 20 etiquetas de cada tipo
            top_user_tags = dict(user_tag_freq.most_common(20))
            top_ai_tags = dict(ai_tag_freq.most_common(20))
            
            # 1. Barras comparativas: User vs AI tags
            fig1 = go.Figure()
            
            fig1.add_trace(go.Bar(
                x=list(top_user_tags.keys()),
                y=list(top_user_tags.values()),
                name='Etiquetas Usuario',
                marker_color='#3498db'
            ))
            
            fig1.add_trace(go.Bar(
                x=list(top_ai_tags.keys()),
                y=list(top_ai_tags.values()),
                name='Etiquetas IA',
                marker_color='#e74c3c'
            ))
            
            fig1.update_layout(
                title="🏷️ Top Etiquetas: Usuario vs IA",
                xaxis_title="Etiqueta",
                yaxis_title="Frecuencia",
                barmode='group',
                xaxis_tickangle=45
            )
            
            # 2. Word cloud data (para visualización en frontend)
            wordcloud_data = {
                "user_tags": [{"text": tag, "value": freq} for tag, freq in user_tag_freq.most_common(50)],
                "ai_tags": [{"text": tag, "value": freq} for tag, freq in ai_tag_freq.most_common(50)]
            }
            
            # 3. Gráfico de barras horizontales
            combined_tags = {}
            for tag in set(list(top_user_tags.keys()) + list(top_ai_tags.keys())):
                combined_tags[tag] = {
                    'user': top_user_tags.get(tag, 0),
                    'ai': top_ai_tags.get(tag, 0)
                }
            
            # Convertir a DataFrame para gráfico
            tags_df = pd.DataFrame(combined_tags).T.reset_index()
            tags_df.columns = ['tag', 'user', 'ai']
            tags_df['total'] = tags_df['user'] + tags_df['ai']
            tags_df = tags_df.sort_values('total', ascending=True).tail(15)
            
            fig3 = go.Figure()
            
            fig3.add_trace(go.Bar(
                y=tags_df['tag'],
                x=tags_df['user'],
                name='Usuario',
                orientation='h',
                marker_color='#3498db'
            ))
            
            fig3.add_trace(go.Bar(
                y=tags_df['tag'],
                x=tags_df['ai'],
                name='IA',
                orientation='h',
                marker_color='#e74c3c'
            ))
            
            fig3.update_layout(
                title="📋 Top 15 Etiquetas Combinadas",
                xaxis_title="Frecuencia",
                yaxis_title="Etiqueta",
                barmode='stack',
                height=500
            )
            
            charts = {
                "tag_comparison": fig1.to_html(full_html=False, include_plotlyjs='cdn'),
                "horizontal_tags": fig3.to_html(full_html=False, include_plotlyjs='cdn')
            }
            
            return {
                "success": True,
                "charts": charts,
                "wordcloud_data": wordcloud_data,
                "statistics": {
                    "total_user_tags": len(user_tag_freq),
                    "total_ai_tags": len(ai_tag_freq),
                    "unique_tags_both": len(set(all_user_tags).union(set(all_ai_tags))),
                    "most_common_user_tag": max(user_tag_freq, key=user_tag_freq.get) if user_tag_freq else None,
                    "most_common_ai_tag": max(ai_tag_freq, key=ai_tag_freq.get) if ai_tag_freq else None
                },
                "metadata": {
                    "analysis_date": datetime.now().isoformat(),
                    "document_count": len(data),
                    "limit_applied": 300
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error en análisis de etiquetas: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
  
    @staticmethod
    async def create_performance_radar():
        """Radar chart comparativo de rendimiento - Retorna JSON"""
        try:
            # Obtener métricas de todos los modelos
            classification = await ModelPerformanceEvaluator.evaluate_image_classification()
            sentiment = await ModelPerformanceEvaluator.evaluate_sentiment_analysis()
            objects = await ModelPerformanceEvaluator.evaluate_object_detection()
            
            # Extraer scores normalizados (0-1)
            scores = {}
            
            # Clasificación
            if 'error' not in classification and 'metrics' in classification:
                scores['classification'] = {
                    'Precision': classification['metrics'].get('avg_precision', 0),
                    'Recall': classification['metrics'].get('avg_recall', 0),
                    'F1 Score': classification['metrics'].get('avg_f1_score', 0)
                }
            
            # Sentimiento
            if 'error' not in sentiment and 'summary_stats' in sentiment:
                sentiment_score = (sentiment['summary_stats'].get('mean_sentiment', 0) + 1) / 2
                scores['sentiment'] = {
                    'Precisión': sentiment_score,
                    'Correlación': min(abs(sentiment.get('correlation_metrics', {}).get('sentiment_likes_correlation', 0)), 1),
                    'Estabilidad': 1 - min(sentiment['summary_stats'].get('std_sentiment', 1), 1)
                }
            
            # Detección de objetos
            if 'error' not in objects and 'object_statistics' in objects:
                obj_score = min(objects['object_statistics'].get('avg_objects_per_image', 0) / 5, 1)
                scores['object_detection'] = {
                    'Cobertura': obj_score,
                    'Diversidad': min(objects['object_statistics'].get('total_unique_objects', 0) / 50, 1),
                    'Precisión': objects['object_statistics'].get('avg_confidence', 0.7)  # Usar confianza real
                }
            
            if not scores:
                return {
                    "success": False,
                    "error": "No hay métricas para radar chart",
                    "data": None
                }
            
            # Definir métricas comunes para comparación
            all_metrics = ['Precisión', 'Recall', 'F1 Score', 'Cobertura', 'Diversidad', 'Correlación', 'Estabilidad']
            
            # Preparar datos para radar chart (formato JSON)
            radar_data = {
                "labels": ['Precisión', 'Recall', 'F1 Score', 'Cobertura', 'Diversidad'],
                "datasets": []
            }
            
            # Colores para cada modelo
            model_colors = {
                'classification': 'rgba(54, 162, 235, 0.6)',
                'sentiment': 'rgba(255, 99, 132, 0.6)',
                'object_detection': 'rgba(75, 192, 192, 0.6)'
            }
            
            model_names = {
                'classification': 'Clasificación de Imágenes',
                'sentiment': 'Análisis de Sentimiento',
                'object_detection': 'Detección de Objetos'
            }
            
            # Construir datasets para cada modelo
            for model_key, model_scores in scores.items():
                # Mapear las métricas del modelo a las métricas del radar
                values = []
                for radar_metric in radar_data["labels"]:
                    # Buscar la métrica correspondiente (con diferentes nombres posibles)
                    metric_value = 0
                    for score_key, score_value in model_scores.items():
                        if radar_metric.lower() in score_key.lower() or score_key.lower() in radar_metric.lower():
                            metric_value = score_value
                            break
                    values.append(round(metric_value, 3))
                
                radar_data["datasets"].append({
                    "label": model_names.get(model_key, model_key.replace('_', ' ').title()),
                    "data": values,
                    "backgroundColor": model_colors.get(model_key, 'rgba(199, 199, 199, 0.6)'),
                    "borderColor": model_colors.get(model_key, 'rgba(199, 199, 199, 1)'),
                    "borderWidth": 2,
                    "pointBackgroundColor": model_colors.get(model_key, 'rgba(199, 199, 199, 1)')
                })
            
            # Calcular estadísticas comparativas
            comparison_stats = {
                "model_count": len(scores),
                "average_scores": {},
                "best_model_per_metric": {},
                "overall_rankings": []
            }
            
            # Calcular promedios por métrica
            for radar_metric in radar_data["labels"]:
                metric_values = []
                for dataset in radar_data["datasets"]:
                    idx = radar_data["labels"].index(radar_metric)
                    metric_values.append(dataset["data"][idx])
                
                if metric_values:
                    comparison_stats["average_scores"][radar_metric] = round(sum(metric_values) / len(metric_values), 3)
                    
                    # Encontrar el mejor modelo para esta métrica
                    best_idx = metric_values.index(max(metric_values))
                    comparison_stats["best_model_per_metric"][radar_metric] = {
                        "model": radar_data["datasets"][best_idx]["label"],
                        "score": max(metric_values)
                    }
            
            # Calificar cada modelo (puntuación global)
            for dataset in radar_data["datasets"]:
                avg_score = round(sum(dataset["data"]) / len(dataset["data"]), 3)
                
                # Determinar categoría basada en puntuación
                if avg_score >= 0.8:
                    category = "Excelente"
                elif avg_score >= 0.6:
                    category = "Bueno"
                elif avg_score >= 0.4:
                    category = "Aceptable"
                else:
                    category = "Necesita Mejora"
                
                comparison_stats["overall_rankings"].append({
                    "model": dataset["label"],
                    "average_score": avg_score,
                    "category": category,
                    "strengths": [],
                    "weaknesses": []
                })
                
                # Identificar fortalezas y debilidades
                for i, metric in enumerate(radar_data["labels"]):
                    score = dataset["data"][i]
                    if score >= 0.7:
                        comparison_stats["overall_rankings"][-1]["strengths"].append(f"{metric}: {score}")
                    elif score <= 0.3:
                        comparison_stats["overall_rankings"][-1]["weaknesses"].append(f"{metric}: {score}")
            
            # Ordenar modelos por puntuación
            comparison_stats["overall_rankings"].sort(key=lambda x: x["average_score"], reverse=True)
            
            # Datos crudos para análisis adicional
            raw_metrics = {
                "classification_metrics": classification if 'error' not in classification else {},
                "sentiment_metrics": sentiment if 'error' not in sentiment else {},
                "object_detection_metrics": objects if 'error' not in objects else {},
                "normalized_scores": scores
            }
            
            # Recomendaciones basadas en el análisis
            recommendations = []
            if comparison_stats["overall_rankings"]:
                best_model = comparison_stats["overall_rankings"][0]
                worst_model = comparison_stats["overall_rankings"][-1]
                
                recommendations.append({
                    "type": "fortaleza",
                    "message": f"El modelo {best_model['model']} tiene el mejor rendimiento general ({best_model['average_score']})"
                })
                
                recommendations.append({
                    "type": "mejora",
                    "message": f"El modelo {worst_model['model']} necesita atención, especialmente en: {', '.join(worst_model['weaknesses'][:2]) if worst_model['weaknesses'] else 'todas las métricas'}"
                })
                
                # Identificar métricas que necesitan mejora general
                for metric, avg in comparison_stats["average_scores"].items():
                    if avg < 0.5:
                        recommendations.append({
                            "type": "alerta",
                            "message": f"La métrica '{metric}' tiene un promedio bajo ({avg}) en todos los modelos"
                        })
            
            return {
                "success": True,
                "radar_chart_data": radar_data,
                "comparison_statistics": comparison_stats,
                "recommendations": recommendations,
                "raw_metrics": raw_metrics,
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "models_evaluated": list(scores.keys()),
                    "metrics_used": radar_data["labels"],
                    "normalization_range": "0-1"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error creando radar chart: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }



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
        
        # Reporte en formato markdown
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
    
    

@metrics_router.get("/engagement")
async def get_engagement_metrics():
    """Métricas de engagement de las imágenes"""
    try:
        return await ModelPerformanceEvaluator.create_engagement_analysis()
    except Exception as e:
        raise HTTPException(500, f"Error en engagement: {str(e)}")

@metrics_router.get("/temporal")
async def get_temporal_metrics():
    """Análisis temporal de las imágenes"""
    try:
        return await ModelPerformanceEvaluator.create_temporal_analysis()
    except Exception as e:
        raise HTTPException(500, f"Error en análisis temporal: {str(e)}")

@metrics_router.get("/categories")
async def get_category_metrics():
    """Análisis detallado por categorías"""
    try:
        return await ModelPerformanceEvaluator.create_category_analysis()
    except Exception as e:
        raise HTTPException(500, f"Error en análisis de categorías: {str(e)}")

@metrics_router.get("/tags")
async def get_tag_metrics():
    """Análisis de etiquetas y palabras clave"""
    try:
        return await ModelPerformanceEvaluator.create_tag_analysis()
    except Exception as e:
        raise HTTPException(500, f"Error en análisis de etiquetas: {str(e)}")

@metrics_router.get("/performance-radar")
async def get_performance_radar():
    """Radar chart comparativo de rendimiento"""
    try:
        return await ModelPerformanceEvaluator.create_performance_radar()
    except Exception as e:
        raise HTTPException(500, f"Error en radar chart: {str(e)}")

@metrics_router.get("/all-charts")
async def get_all_charts():
    """Obtiene todas las gráficas disponibles"""
    try:
        results = await asyncio.gather(
            ModelPerformanceEvaluator.create_engagement_analysis(),
            ModelPerformanceEvaluator.create_temporal_analysis(),
            ModelPerformanceEvaluator.create_category_analysis(),
            ModelPerformanceEvaluator.create_tag_analysis(),
            ModelPerformanceEvaluator.create_performance_radar(),
            return_exceptions=True
        )
        
        charts = {
            "engagement": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
            "temporal": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
            "categories": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
            "tags": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
            "performance_radar": results[4] if not isinstance(results[4], Exception) else {"error": str(results[4])}
        }
        
        # Contar gráficas exitosas
        successful = sum(1 for v in charts.values() if 'error' not in v)
        
        return {
            "charts": charts,
            "summary": {
                "total_charts": len(charts),
                "successful_charts": successful,
                "failed_charts": len(charts) - successful
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error obteniendo todas las gráficas: {str(e)}")