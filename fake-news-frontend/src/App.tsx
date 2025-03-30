import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FiAlertTriangle, FiCheckCircle, FiTrash2, FiRefreshCw, FiUpload, FiInfo } from 'react-icons/fi';

interface NewsItem {
  text: string;
  source?: string;
  date?: string;
}

interface PredictionResult {
  prediction: number;
  probability: number;
  explanation?: string;
}

interface TrainingDataItem {
  text: string;
  label: number;
  source?: string;
  date?: string;
}

interface RetrainResults {
  strategy_used: string;
  new_samples: number;
  training_time: number;
  new_version?: string;
  metrics: {
    accuracy: number;
    precision_0: number;
    recall_0: number;
    f1_0: number;
    precision_1: number;
    recall_1: number;
    f1_1: number;
    confusion_matrix: number[][];
  };
  error?: string;
}

interface ModelInfo {
  version: string;
  type: string;
  last_retrain: string;
}

const API_URL = 'http://localhost:8000';

const App: React.FC = () => {
  const [inputText, setInputText] = useState<string>('');
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'analyze' | 'train'>('analyze');
  const [trainingData, setTrainingData] = useState<TrainingDataItem[]>([]);
  const [newTrainingText, setNewTrainingText] = useState<string>('');
  const [newTrainingLabel, setNewTrainingLabel] = useState<number>(0);
  const [retrainStrategy, setRetrainStrategy] = useState<string>('incremental');
  const [isRetraining, setIsRetraining] = useState<boolean>(false);
  const [retrainResults, setRetrainResults] = useState<RetrainResults | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  // Cargar resultados previos y info del modelo
  useEffect(() => {
    const savedResults = localStorage.getItem('lastRetrainResults');
    if (savedResults) {
      setRetrainResults(JSON.parse(savedResults));
    }
    fetchModelInfo();
  }, []);

  // Guardar resultados cuando cambian
  useEffect(() => {
    if (retrainResults) {
      localStorage.setItem('lastRetrainResults', JSON.stringify(retrainResults));
    }
  }, [retrainResults]);

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API_URL}/model_info`);
      setModelInfo(response.data);
    } catch (error) {
      console.error('Error fetching model info:', error);
    }
  };

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;

    setIsAnalyzing(true);
    try {
      const items = inputText.split('\n\n').filter(t => t.trim()).map(text => ({ text }));
      setNewsItems(items);

      const response = await axios.post(`${API_URL}/predict`, {
        news_items: items
      });

      setPredictions(response.data.results);
    } catch (error) {
      console.error('Error analyzing text:', error);
      alert('Error analizando el texto. Por favor intente nuevamente.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAddTrainingData = () => {
    if (!newTrainingText.trim()) return;

    setTrainingData(prev => [
      ...prev,
      { text: newTrainingText, label: newTrainingLabel }
    ]);
    setNewTrainingText('');
  };

  const handleRetrain = async () => {
    if (trainingData.length === 0) return;

    setIsRetraining(true);
    try {
      const response = await axios.post(`${API_URL}/retrain`, {
        data: trainingData,
        strategy: retrainStrategy
      });

      // Validar y extraer métricas de forma segura
      const metrics = {
        accuracy: Number(response.data.metrics?.accuracy) || 0,
        precision_0: Number(response.data.metrics?.['0']?.precision) || 0,
        recall_0: Number(response.data.metrics?.['0']?.recall) || 0,
        f1_0: Number(response.data.metrics?.['0']?.['f1-score']) || 0,
        precision_1: Number(response.data.metrics?.['1']?.precision) || 0,
        recall_1: Number(response.data.metrics?.['1']?.recall) || 0,
        f1_1: Number(response.data.metrics?.['1']?.['f1-score']) || 0,
        confusion_matrix: response.data.confusion_matrix || [[0, 0], [0, 0]]
      };

      setRetrainResults({
        strategy_used: response.data.strategy_used || retrainStrategy,
        new_samples: response.data.new_samples || trainingData.length,
        training_time: response.data.training_time || 0,
        new_version: response.data.model_version || modelInfo?.version,
        metrics: metrics
      });
      
      setTrainingData([]);
      await fetchModelInfo();
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 
                         error.message || 
                         'Error desconocido durante el reentrenamiento';
      console.error('Error:', errorMessage);
      
      setRetrainResults({
        strategy_used: retrainStrategy,
        new_samples: 0,
        training_time: 0,
        new_version: modelInfo?.version,
        metrics: {
          accuracy: 0,
          precision_0: 0,
          recall_0: 0,
          f1_0: 0,
          precision_1: 0,
          recall_1: 0,
          f1_1: 0,
          confusion_matrix: [[0, 0], [0, 0]]
        },
        error: errorMessage
      });
    } finally {
      setIsRetraining(false);
    }
  };

  const getPredictionColor = (prediction: number, probability: number) => {
    if (prediction === 1) {
      return `rgba(239, 68, 68, ${0.1 + probability * 0.6})`;
    } else {
      return `rgba(16, 185, 129, ${0.1 + (1 - probability) * 0.6})`;
    }
  };

  const getPredictionTextColor = (prediction: number) => {
    return prediction === 1 ? 'text-red-400' : 'text-green-400';
  };

  const getPredictionIcon = (prediction: number) => {
    return prediction === 1 ? 
      <FiAlertTriangle className="inline mr-2" /> : 
      <FiCheckCircle className="inline mr-2" />;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-gray-100">
      <header className="bg-gray-800 border-b border-gray-700 p-4 shadow-lg">
        <div className="container mx-auto flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <div className="bg-blue-600 p-2 rounded-lg mr-3">
              <FiInfo className="text-white text-xl" />
            </div>
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-blue-200">
              Detector de Noticias Falsas
            </h1>
          </div>
          <div className="flex flex-col text-sm text-gray-300 text-center md:text-right">
            <span className="font-mono bg-gray-700 px-2 py-1 rounded">
              Modelo: {modelInfo?.version || '1'} | {modelInfo?.type?.split('.')?.pop() || 'NLP'}
            </span>
            <span className="text-xs mt-1">
              Última actualización: {modelInfo?.last_retrain ? new Date(modelInfo.last_retrain).toLocaleString() : 'N/A'}
            </span>
          </div>
        </div>
      </header>

      <main className="container mx-auto p-4">
        <div className="flex mb-6 border-b border-gray-700">
          <button
            className={`px-4 py-3 font-medium flex items-center ${activeTab === 'analyze' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-200'}`}
            onClick={() => setActiveTab('analyze')}
          >
            <FiUpload className="mr-2" /> Analizar Noticias
          </button>
          <button
            className={`px-4 py-3 font-medium flex items-center ${activeTab === 'train' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-200'}`}
            onClick={() => setActiveTab('train')}
          >
            <FiRefreshCw className="mr-2" /> Entrenar Modelo
          </button>
        </div>

        {activeTab === 'analyze' ? (
          <div className="bg-gray-800 rounded-xl shadow-xl p-6 mb-8 border border-gray-700">
            <h2 className="text-xl font-semibold mb-6 text-blue-300">Analizar Contenido de Noticias</h2>
            <div className="mb-6">
              <label className="block text-gray-300 mb-3 font-medium" htmlFor="news-text">
                Ingrese texto de noticias (separe artículos con líneas vacías):
              </label>
              <textarea
                id="news-text"
                className="w-full h-64 p-4 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-200 placeholder-gray-500"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Pegue el texto del artículo aquí..."
              />
            </div>
            <div className="flex justify-end">
              <button
                className="bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white font-bold py-3 px-8 rounded-lg transition duration-200 disabled:opacity-50 flex items-center"
                onClick={handleAnalyze}
                disabled={isAnalyzing || !inputText.trim()}
              >
                {isAnalyzing ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analizando...
                  </>
                ) : (
                  'Analizar'
                )}
              </button>
            </div>

            {predictions.length > 0 && (
              <div className="mt-10">
                <h3 className="text-lg font-semibold mb-6 text-blue-300">Resultados del Análisis</h3>
                <div className="space-y-6">
                  {newsItems.map((item, index) => (
                    <div key={index} className="rounded-lg overflow-hidden border border-gray-700">
                      <div 
                        className="p-4 font-medium flex justify-between items-center"
                        style={{ 
                          backgroundColor: getPredictionColor(
                            predictions[index].prediction, 
                            predictions[index].probability
                          ) 
                        }}
                      >
                        <div className={`flex items-center ${getPredictionTextColor(predictions[index].prediction)}`}>
                          {getPredictionIcon(predictions[index].prediction)}
                          {predictions[index].prediction === 1 ? (
                            <span>Probablemente Falsa ({Math.round(predictions[index].probability * 100)}% confianza)</span>
                          ) : (
                            <span>Probablemente Real ({Math.round((1 - predictions[index].probability) * 100)}% confianza)</span>
                          )}
                        </div>
                        <span className="text-xs bg-black bg-opacity-20 px-2 py-1 rounded">
                          Artículo {index + 1} de {newsItems.length}
                        </span>
                      </div>
                      <div className="p-5 bg-gray-700">
                        <div className="mb-4">
                          <h4 className="text-sm font-medium text-gray-400 mb-2">Vista previa:</h4>
                          <p className="text-gray-200">{item.text.slice(0, 300)}{item.text.length > 300 ? '...' : ''}</p>
                        </div>
                        
                        {predictions[index].explanation && (
                          <div className="mb-4">
                            <h4 className="text-sm font-medium text-gray-400 mb-2">Indicadores clave:</h4>
                            <p className="text-gray-300">{predictions[index].explanation}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="bg-gray-800 rounded-xl shadow-xl p-6 mb-8 border border-gray-700">
            <h2 className="text-xl font-semibold mb-6 text-blue-300">Entrenamiento del Modelo</h2>
            
            <div className="mb-8 p-5 bg-gray-700 rounded-lg border border-gray-600">
              <h3 className="font-medium text-blue-300 mb-3">Estrategias de Reentrenamiento</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button
                  className={`p-4 border rounded-lg transition-all text-left ${retrainStrategy === 'incremental' ? 'border-blue-400 bg-gray-600 shadow-lg ring-2 ring-blue-500' : 'border-gray-600 hover:border-gray-500'}`}
                  onClick={() => setRetrainStrategy('incremental')}
                >
                  <h4 className="font-medium flex items-center">
                    <span className="w-3 h-3 bg-blue-400 rounded-full mr-2"></span>
                    Incremental
                  </h4>
                  <p className="text-sm text-gray-400 mt-2">
                    Actualiza el modelo solo con nuevos datos. Rápido pero puede causar olvido catastrófico.
                  </p>
                  <div className="mt-3 text-xs text-gray-500">
                    <span className="bg-gray-800 px-2 py-1 rounded">Recomendado para pequeñas actualizaciones</span>
                  </div>
                </button>
                <button
                  className={`p-4 border rounded-lg transition-all text-left ${retrainStrategy === 'partial' ? 'border-blue-400 bg-gray-600 shadow-lg ring-2 ring-blue-500' : 'border-gray-600 hover:border-gray-500'}`}
                  onClick={() => setRetrainStrategy('partial')}
                >
                  <h4 className="font-medium flex items-center">
                    <span className="w-3 h-3 bg-purple-400 rounded-full mr-2"></span>
                    Parcial
                  </h4>
                  <p className="text-sm text-gray-400 mt-2">
                    Combina nuevos datos con muestras de datos antiguos. Enfoque equilibrado.
                  </p>
                  <div className="mt-3 text-xs text-gray-500">
                    <span className="bg-gray-800 px-2 py-1 rounded">Recomendado para actualizaciones medianas</span>
                  </div>
                </button>
                <button
                  className={`p-4 border rounded-lg transition-all text-left ${retrainStrategy === 'full' ? 'border-blue-400 bg-gray-600 shadow-lg ring-2 ring-blue-500' : 'border-gray-600 hover:border-gray-500'}`}
                  onClick={() => setRetrainStrategy('full')}
                >
                  <h4 className="font-medium flex items-center">
                    <span className="w-3 h-3 bg-green-400 rounded-full mr-2"></span>
                    Completo
                  </h4>
                  <p className="text-sm text-gray-400 mt-2">
                    Reentrena desde cero con todos los datos disponibles. Más preciso pero intensivo en recursos.
                  </p>
                  <div className="mt-3 text-xs text-gray-500">
                    <span className="bg-gray-800 px-2 py-1 rounded">Recomendado para actualizaciones mayores</span>
                  </div>
                </button>
              </div>
            </div>

            <div className="mb-8">
              <h3 className="font-medium text-blue-300 mb-4">Agregar Datos de Entrenamiento</h3>
              
              <div className="flex flex-col md:flex-row gap-4 mb-6">
                <div className="flex-1">
                  <textarea
                    className="w-full h-40 p-4 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-200 placeholder-gray-500"
                    value={newTrainingText}
                    onChange={(e) => setNewTrainingText(e.target.value)}
                    placeholder="Ingrese texto para entrenamiento..."
                  />
                </div>
                <div className="w-full md:w-64">
                  <label className="block text-gray-300 mb-2">Etiqueta:</label>
                  <select
                    className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={newTrainingLabel}
                    onChange={(e) => setNewTrainingLabel(Number(e.target.value))}
                  >
                    <option value={0}>Noticia Real</option>
                    <option value={1}>Noticia Falsa</option>
                  </select>
                  <button
                    className="mt-4 w-full bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white font-bold py-3 px-4 rounded-lg transition duration-200 disabled:opacity-50"
                    onClick={handleAddTrainingData}
                    disabled={!newTrainingText.trim()}
                  >
                    Agregar al Conjunto
                  </button>
                </div>
              </div>
            </div>

            {trainingData.length > 0 && (
              <div className="mb-8">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="font-medium text-blue-300">
                    Conjunto de Entrenamiento ({trainingData.length} elementos)
                  </h3>
                  <button 
                    className="text-red-400 hover:text-red-300 text-sm flex items-center"
                    onClick={() => setTrainingData([])}
                  >
                    <FiTrash2 className="mr-1" /> Limpiar Todo
                  </button>
                </div>
                <div className="border border-gray-600 rounded-lg overflow-hidden">
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-600">
                      <thead className="bg-gray-700">
                        <tr>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Vista previa</th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Etiqueta</th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Acciones</th>
                        </tr>
                      </thead>
                      <tbody className="bg-gray-800 divide-y divide-gray-700">
                        {trainingData.map((item, index) => (
                          <tr key={index} className="hover:bg-gray-750">
                            <td className="px-4 py-3 text-sm text-gray-300 max-w-xs truncate">
                              {item.text.slice(0, 100)}{item.text.length > 100 ? '...' : ''}
                            </td>
                            <td className="px-4 py-3 text-sm">
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${item.label === 1 ? 'bg-red-900 text-red-200' : 'bg-green-900 text-green-200'}`}>
                                {item.label === 1 ? 'Falsa' : 'Real'}
                              </span>
                            </td>
                            <td className="px-4 py-3 text-sm">
                              <button 
                                className="text-red-400 hover:text-red-300 flex items-center"
                                onClick={() => setTrainingData(prev => prev.filter((_, i) => i !== index))}
                              >
                                <FiTrash2 className="mr-1" /> Eliminar
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            <div className="flex justify-end mb-6">
              <button
                className="bg-gradient-to-r from-green-600 to-green-500 hover:from-green-500 hover:to-green-400 text-white font-bold py-3 px-8 rounded-lg transition duration-200 disabled:opacity-50 flex items-center"
                onClick={handleRetrain}
                disabled={isRetraining || trainingData.length === 0}
              >
                {isRetraining ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Reentrenando...
                  </>
                ) : (
                  'Reentrenar Modelo'
                )}
              </button>
            </div>

            {retrainResults && (
              <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                <h3 className="text-lg font-semibold mb-3 text-blue-300">
                  {retrainResults.error ? 'Error en Reentrenamiento' : 'Resultados del Reentrenamiento'}
                </h3>
                
                {retrainResults.error ? (
                  <div className="bg-red-900 text-red-200 p-3 rounded mb-3">
                    Error: {retrainResults.error}
                  </div>
                ) : (
                  <>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                      <div className="bg-blue-900 bg-opacity-30 p-3 rounded border border-blue-700">
                        <h4 className="text-xs font-medium text-gray-400 mb-1">Precisión Global</h4>
                        <p className="text-xl font-bold">{(retrainResults.metrics.accuracy * 100).toFixed(1)}%</p>
                      </div>
                      <div className="bg-green-900 bg-opacity-30 p-3 rounded border border-green-700">
                        <h4 className="text-xs font-medium text-gray-400 mb-1">Precisión (Reales)</h4>
                        <p className="text-xl font-bold">{(retrainResults.metrics.precision_0 * 100).toFixed(1)}%</p>
                      </div>
                      <div className="bg-red-900 bg-opacity-30 p-3 rounded border border-red-700">
                        <h4 className="text-xs font-medium text-gray-400 mb-1">Precisión (Falsas)</h4>
                        <p className="text-xl font-bold">{(retrainResults.metrics.precision_1 * 100).toFixed(1)}%</p>
                      </div>
                      <div className="bg-purple-900 bg-opacity-30 p-3 rounded border border-purple-700">
                        <h4 className="text-xs font-medium text-gray-400 mb-1">Tiempo</h4>
                        <p className="text-xl font-bold">{retrainResults.training_time.toFixed(2)}s</p>
                      </div>
                    </div>

                    <div className="mt-4">
                      <h4 className="text-sm font-medium text-gray-400 mb-2">Matriz de Confusión:</h4>
                      <div className="bg-gray-800 p-3 rounded">
                        <table className="w-full">
                          <tbody>
                            <tr>
                              <td className="text-center p-2 border border-gray-600">Verdaderos Negativos</td>
                              <td className="text-center p-2 border border-gray-600">Falsos Positivos</td>
                            </tr>
                            <tr>
                              <td className="text-center p-2 border border-gray-600">{retrainResults.metrics.confusion_matrix[0][0]}</td>
                              <td className="text-center p-2 border border-gray-600">{retrainResults.metrics.confusion_matrix[0][1]}</td>
                            </tr>
                            <tr>
                              <td className="text-center p-2 border border-gray-600">Falsos Negativos</td>
                              <td className="text-center p-2 border border-gray-600">Verdaderos Positivos</td>
                            </tr>
                            <tr>
                              <td className="text-center p-2 border border-gray-600">{retrainResults.metrics.confusion_matrix[1][0]}</td>
                              <td className="text-center p-2 border border-gray-600">{retrainResults.metrics.confusion_matrix[1][1]}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                )}

                <div className="text-sm mt-3">
                  <p><span className="text-gray-400">Estrategia:</span> <span className="capitalize">{retrainResults.strategy_used}</span></p>
                  <p><span className="text-gray-400">Muestras:</span> {retrainResults.new_samples} nuevas</p>
                  <p><span className="text-gray-400">Versión modelo:</span> v{retrainResults.new_version || modelInfo?.version}</p>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="bg-gray-800 p-6 border-t border-gray-700 mt-12">
        <div className="container mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0">
              <h3 className="text-lg font-medium text-gray-300">Detector de Noticias Falsas</h3>
              <p className="text-sm text-gray-500">Sistema Avanzado de Detección</p>
            </div>
            <div className="text-center md:text-right">
              <p className="text-sm text-gray-400">Universidad de los Andes - ISIS 3301</p>
              <p className="text-xs text-gray-600 mt-1">
                Versión del modelo: {modelInfo?.version || '1.0'} | Última actualización: {'29/03/2026'}
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;