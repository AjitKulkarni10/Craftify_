import React, { useState, useEffect } from 'react';
import { BookOpen, Sparkles, Brain, Target, Users, Award, ChevronRight, Lock, Mail, CheckCircle, BookOpenCheck, ListChecks, FileText, ArrowLeft, Clock, Code, Lightbulb, Globe, BookText } from 'lucide-react';
import axios from 'axios';

// Define the Course type
interface Course {
  courseTitle: string;
  courseOverview: string;
  modules: {
    moduleTitle: string;
    moduleOverview: string;
    keyTopics: string[];
    detailedContent: {
      concept: string;
      explanation: string;
      example: string;
      realWorldRelevance: string;
    }[];
    videoUrl?: string;  // Add video URL field
  }[];
}

interface quiz {
  quizTitle: string;
  totalQuestions: number;
  timeLimit: number;
  questions: Array<{
    id: string;
    question: string;
    options: Array<{
      id: string;
      text: string;
    }>;
    correctOptionId: string;
    points: number;
  }>;
}

interface MindMapNode {
  id: string;
  type: string;
  text: string;
  children?: MindMapNode[];
}

interface CourseDisplayProps {
  course: Course | null;
  onBack: () => void;
}

interface ModuleDetailProps {
  module: Course['modules'][0];
  onBack: () => void;
}

interface CourseHistory extends Course {
  createdAt: string;
  id: string;
}

interface QuizDisplayProps {
  quiz: quiz;
  onBack: () => void;
}

function LoginPage({ onLogin }: { onLogin: () => void }) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <BookOpen className="mx-auto h-12 w-12 text-primary-600" />
          <h2 className="mt-6 text-3xl font-extrabold text-gray-900">Craftify</h2>
          <p className="mt-2 text-sm text-gray-600">Sign in to your account</p>
        </div>
        <form className="mt-8 space-y-6" onSubmit={(e) => { e.preventDefault(); onLogin(); }}>
          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <label htmlFor="email-address" className="sr-only">Email address</label>
              <div className="flex items-center">
                <Mail className="h-5 w-5 text-gray-400 absolute ml-3" />
                <input
                  id="email-address"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  className="appearance-none rounded-none relative block w-full px-12 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 sm:text-sm"
                  placeholder="Email address"
                />
              </div>
            </div>
            <div>
              <label htmlFor="password" className="sr-only">Password</label>
              <div className="flex items-center">
                <Lock className="h-5 w-5 text-gray-400 absolute ml-3" />
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  required
                  className="appearance-none rounded-none relative block w-full px-12 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 sm:text-sm"
                  placeholder="Password"
                />
              </div>
            </div>
          </div>

          <div>
            <button
              type="submit"
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
            >
              Sign in
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function ModuleDetail({ module, onBack }: ModuleDetailProps) {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [isLoadingVideo, setIsLoadingVideo] = useState(false);
  const [showTranscript, setShowTranscript] = useState(false);

  useEffect(() => {
    const fetchVideo = async () => {
      if (!module.videoUrl) {
        setIsLoadingVideo(true);
        try {
          const response = await axios.post('http://127.0.0.1:8001/get-video/', {
            topic: module.moduleTitle
          });
          setVideoUrl(response.data.video_url);
          setTranscript(response.data.transcript); // 🆕 Save transcript
        } catch (error) {
          console.error('Error fetching video:', error);
        } finally {
          setIsLoadingVideo(false);
        }
      } else {
        setVideoUrl(module.videoUrl);
      }
    };

    fetchVideo();
  }, [module]);

  return (
    <div className="bg-white rounded-lg shadow-lg">
      <div className="bg-primary-600 text-white rounded-t-lg p-8">
        <button
          onClick={onBack}
          className="flex items-center text-white mb-6 hover:opacity-80"
        >
          <ArrowLeft className="h-5 w-5 mr-2" />
          Back to Modules
        </button>
        <h1 className="text-3xl font-bold mb-4">{module.moduleTitle}</h1>
        <p className="text-lg mb-4">{module.moduleOverview}</p>
      </div>

      <div className="p-8">
        {/* Video Section */}
        <div className="mb-8">
          <h2 className="text-xl font-bold mb-4">Related Video</h2>
          {isLoadingVideo ? (
            <div className="flex items-center justify-center h-64 bg-gray-100 rounded-lg">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
            </div>
          ) : videoUrl ? (
            <div>
              <div className="aspect-w-16 aspect-h-9">
                <iframe
                  src={videoUrl.replace('watch?v=', 'embed/')}
                  title="YouTube video player"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  className="w-full h-96 rounded-lg"
                ></iframe>
              </div>

              {/* 🆕 Transcript Toggle Button */}
              {transcript && (
                <div className="mt-4">
                  <button
                    onClick={() => setShowTranscript(!showTranscript)}
                    className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition"
                  >
                    {showTranscript ? 'Hide Transcript' : 'Show Transcript'}
                  </button>
                </div>
              )}

              {/* 🆕 Transcript Content */}
              {showTranscript && transcript && (
                <div className="mt-4 p-4 bg-gray-100 rounded-lg text-sm whitespace-pre-wrap">
                  {transcript}
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8 bg-gray-50 rounded-lg">
              <p className="text-gray-500">No video available for this module</p>
            </div>
          )}
        </div>

        {/* Key Topics */}
        <div className="mb-8">
          <h2 className="text-xl font-bold mb-4">Key Topics</h2>
          <ul className="list-disc pl-5 space-y-2">
            {(module.keyTopics || []).map((topic, index) => (
              <li key={index} className="text-gray-700">{topic}</li>
            ))}
          </ul>
        </div>

        {/* Detailed Content */}
        <div className="space-y-8">
          <h2 className="text-xl font-bold mb-4">Detailed Content</h2>
          {(module.detailedContent || []).map((content, index) => (
            <div key={index} className="bg-gray-50 rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-3">{content.concept}</h3>
              <div className="space-y-4">
                <div>
                  <p className="text-gray-700">{content.explanation}</p>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Example:</h4>
                  <p className="text-gray-700">{content.example}</p>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Real-World Relevance:</h4>
                  <p className="text-gray-700">{content.realWorldRelevance}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Add MindMap component
function MindMap({ data, onClose }: { data: MindMapNode, onClose: () => void }) {
  return (
    <div className="mindmap-modal">
      <div className="modal-content">
        <div className="modal-header">
          <h2 className="text-2xl font-bold">Course Mind Map</h2>
          <button
            onClick={onClose}
            className="text-white hover:text-white/80 transition-colors"
            aria-label="Close mindmap"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="mindmap-container">
          <div className="diagram-container">
            {renderMindMapNode(data)}
          </div>
        </div>
      </div>
    </div>
  );
}

function renderMindMapNode(node: MindMapNode) {
  if (node.type === 'root') {
    return (
      <div className="root-node animate-fadeIn">
        <div className="root-card">
          <h3 className="text-xl font-bold text-gray-800">{node.text}</h3>
        </div>
        {node.children && node.children.length > 0 && (
          <>
            <div className="connection-line root-connection" />
            <div className="modules-container">
              {node.children.map((moduleNode, index) => (
                <div key={moduleNode.id} className="module-node animate-slideUp" style={{ animationDelay: `${index * 150}ms` }}>
                  <div className="module-card">
                    <h4 className="text-lg font-semibold text-gray-800">{moduleNode.text}</h4>
                  </div>
                  {moduleNode.children && moduleNode.children.length > 0 && (
                    <>
                      <div className="connection-line module-connection" />
                      <div className="topics-container">
                        {moduleNode.children.map((topicNode, topicIndex) => (
                          <div
                            key={topicNode.id}
                            className="topic-tag animate-flipIn"
                            style={{ animationDelay: `${(index * 150) + (topicIndex * 100)}ms` }}
                          >
                            <div className="topic-card">
                              {topicNode.text}
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    );
  }
  return null;
}

function QuizDisplay({ quiz, onBack }: QuizDisplayProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswers, setSelectedAnswers] = useState<Record<string, string>>({});
  quiz.timeLimit=10;
  const [showResults, setShowResults] = useState(false);
  const [timeLeft, setTimeLeft] = useState(quiz.timeLimit * 60); // Convert minutes to seconds

  useEffect(() => {
    if (timeLeft > 0 && !showResults) {
      const timer = setInterval(() => {
        setTimeLeft(prev => prev - 1);
      }, 1000);
      return () => clearInterval(timer);
    } else if (timeLeft === 0 && !showResults) {
      setShowResults(true);
    }
  }, [timeLeft, showResults]);

  const handleAnswerSelect = (questionId: string, optionId: string) => {
    setSelectedAnswers(prev => ({
      ...prev,
      [questionId]: optionId
    }));
  };

  const calculateScore = () => {
    let totalScore = 0;
    let maxScore = 0;
    quiz.questions.forEach(question => {
      maxScore += question.points;
      if (selectedAnswers[question.id] === question.correctOptionId) {
        totalScore += question.points;
      }
    });
    return { totalScore, maxScore };
  };

  if (showResults) {
    const { totalScore, maxScore } = calculateScore();
    return (
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Quiz Results</h2>
          <div className="text-4xl font-bold text-primary-600 mb-2">
            {totalScore}/{maxScore} points
          </div>
          <p className="text-gray-600 mb-6">
            {((totalScore / maxScore) * 100).toFixed(1)}% correct
          </p>
        </div>

        <div className="space-y-8">
          <h3 className="text-xl font-semibold text-gray-900">Quiz Analysis</h3>
          {quiz.questions.map((question, index) => {
            const userAnswer = selectedAnswers[question.id];
            const isCorrect = userAnswer === question.correctOptionId;
            const correctOption = question.options.find(opt => opt.id === question.correctOptionId);
            const userOption = question.options.find(opt => opt.id === userAnswer);

            return (
              <div 
                key={question.id} 
                className={`p-6 rounded-lg border ${
                  isCorrect ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'
                }`}
              >
                <div className="flex items-start justify-between mb-4">
                  <h4 className="text-lg font-medium text-gray-900">
                    Question {index + 1}
                  </h4>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    isCorrect ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                  }`}>
                    {isCorrect ? 'Correct' : 'Incorrect'}
                  </span>
                </div>

                <p className="text-gray-700 mb-4">{question.question}</p>

                <div className="space-y-3 mb-4">
                  <div>
                    <span className="font-medium text-gray-700">Your Answer: </span>
                    <span className={`${
                      isCorrect ? 'text-green-700' : 'text-red-700'
                    }`}>
                      {userOption ? userOption.text : 'Not answered'}
                    </span>
                  </div>
                  {!isCorrect && (
                    <div>
                      <span className="font-medium text-gray-700">Correct Answer: </span>
                      <span className="text-green-700">{correctOption?.text}</span>
                    </div>
                  )}
                </div>

                <div className="mt-4 flex items-center justify-between text-sm text-gray-500">
                  <span>Points: {question.points}</span>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-8 text-center">
          <button
            onClick={onBack}
            className="px-6 py-3 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
          >
            Back to Course
          </button>
        </div>
      </div>
    );
  }

  const question = quiz.questions[currentQuestion];

  return (
    <div className="bg-white rounded-lg shadow-lg">
      <div className="bg-primary-600 text-white rounded-t-lg p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold">{quiz.quizTitle}</h2>
          <div className="text-lg">
            Time Left: {Math.floor(timeLeft / 60)}:{(timeLeft % 60).toString().padStart(2, '0')}
          </div>
        </div>
      </div>

      <div className="p-6">
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-500">
              Question {currentQuestion + 1} of {quiz.questions.length}
            </span>
            <span className="text-sm font-medium text-primary-600">
              {question.points} points
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-primary-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${((currentQuestion + 1) / quiz.questions.length) * 100}%` }}
            />
          </div>
        </div>

        <div className="mb-8">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">{question.question}</h3>
          <div className="space-y-3">
            {question.options.map(option => (
              <button
                key={option.id}
                onClick={() => handleAnswerSelect(question.id, option.id)}
                className={`w-full text-left p-4 rounded-lg border transition-all ${
                  selectedAnswers[question.id] === option.id
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-200 hover:border-primary-300'
                }`}
              >
                <span className="font-medium">{option.text}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="flex justify-between">
          <button
            onClick={() => setCurrentQuestion(prev => prev - 1)}
            disabled={currentQuestion === 0}
            className="px-4 py-2 text-gray-600 hover:text-gray-900 disabled:opacity-50"
          >
            Previous
          </button>
          {currentQuestion === quiz.questions.length - 1 ? (
            <button
              onClick={() => setShowResults(true)}
              className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
            >
              Submit Quiz
            </button>
          ) : (
            <button
              onClick={() => setCurrentQuestion(prev => prev + 1)}
              className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
            >
              Next
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function CourseDisplay({ course, onBack }: CourseDisplayProps) {
  const [selectedModule, setSelectedModule] = useState<Course['modules'][0] | null>(null);
  const [mindMapData, setMindMapData] = useState<MindMapNode | null>(null);
  const [isGeneratingMindMap, setIsGeneratingMindMap] = useState(false);
  const [quiz, setQuiz] = useState<quiz | null>(null);
  const [isGeneratingQuiz, setIsGeneratingQuiz] = useState(false);

  const generateMindMap = async () => {
    if (!course) return;
    
    setIsGeneratingMindMap(true);
    try {
      const response = await axios.post('http://127.0.0.1:8001/generate-mindmap/', course);
      setMindMapData(response.data);
    } catch (error: any) {
      console.error('Error generating mindmap:', error);
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error('Error response data:', error.response.data);
        console.error('Error response status:', error.response.status);
        alert(`Failed to generate mindmap: ${error.response.data.detail || 'Unknown error'}`);
      } else if (error.request) {
        // The request was made but no response was received
        console.error('Error request:', error.request);
        alert('No response received from server. Please check if the backend is running.');
      } else {
        // Something happened in setting up the request that triggered an Error
        console.error('Error message:', error.message);
        alert(`Error: ${error.message}`);
      }
    } finally {
      setIsGeneratingMindMap(false);
    }
  };

  const generateQuiz = async () => {
    if (!course) return;
    
    setIsGeneratingQuiz(true);
    try {
      const response = await axios.post('http://127.0.0.1:8001/generate-quiz/', {
        topic: course.courseTitle
      });
      console.log('Quiz generation response:', response.data);
      setQuiz(response.data);
    } catch (error: any) {
      console.error('Error generating quiz:', error);
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error('Error response data:', error.response.data);
        console.error('Error response status:', error.response.status);
        alert(`Failed to generate quiz: ${error.response.data.detail || 'Unknown error'}`);
      } else if (error.request) {
        // The request was made but no response was received
        console.error('Error request:', error.request);
        alert('No response received from server. Please check if the backend is running.');
      } else {
        // Something happened in setting up the request that triggered an Error
        console.error('Error message:', error.message);
        alert(`Error: ${error.message}`);
      }
    } finally {
      setIsGeneratingQuiz(false);
    }
  };

  if (!course || !Array.isArray(course.modules) || course.modules.length === 0) {
    return (
      <div className="p-8">
        <p className="text-center text-gray-500 text-lg">No course content available.</p>
      </div>
    );
  }

  if (selectedModule) {
    return <ModuleDetail module={selectedModule} onBack={() => setSelectedModule(null)} />;
  }

  if (quiz) {
    return <QuizDisplay quiz={quiz} onBack={() => setQuiz(null)} />;
  }

  return (
    <div className="bg-white rounded-lg shadow-lg">
      {/* Course Header */}
      <div className="bg-primary-600 text-white rounded-t-lg p-8">
        <div className="flex justify-between items-center mb-6">
          <button
            onClick={onBack}
            className="flex items-center text-white hover:opacity-80"
          >
            <Code className="h-5 w-5 mr-2" />
            Back to Dashboard
          </button>
          <div className="flex space-x-4">
            <button
              onClick={generateMindMap}
              disabled={isGeneratingMindMap}
              className={`flex items-center px-4 py-2 rounded-md ${
                isGeneratingMindMap 
                  ? 'bg-primary-400 cursor-not-allowed' 
                  : 'bg-white text-primary-600 hover:bg-primary-50'
              }`}
            >
              {isGeneratingMindMap ? (
                'Generating...'
              ) : (
                <>
                  <Brain className="h-5 w-5 mr-2" />
                  Generate Mind Map
                </>
              )}
            </button>
            <button
              onClick={generateQuiz}
              disabled={isGeneratingQuiz}
              className={`flex items-center px-4 py-2 rounded-md ${
                isGeneratingQuiz 
                  ? 'bg-primary-400 cursor-not-allowed' 
                  : 'bg-white text-primary-600 hover:bg-primary-50'
              }`}
            >
              {isGeneratingQuiz ? (
                'Generating...'
              ) : (
                <>
                  <CheckCircle className="h-5 w-5 mr-2" />
                  Take Quiz
                </>
              )}
            </button>
          </div>
        </div>
        <h1 className="text-3xl font-bold mb-4">{course.courseTitle} ✨</h1>
        <p className="text-lg mb-4">{course.courseOverview}</p>
      </div>

      {/* Course Modules List */}
      <div className="p-8">
        <h2 className="text-2xl font-bold mb-6">Course Modules 📚</h2>
        <div className="space-y-4">
          {course.modules.map((module, index) => (
            <button
              key={index}
              onClick={() => setSelectedModule(module)}
              className="w-full text-left border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow bg-white"
            >
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-semibold text-gray-900">{module.moduleTitle}</h3>
                  <p className="text-gray-600 mt-2">{module.moduleOverview}</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {(module.keyTopics || []).slice(0, 3).map((topic, topicIndex) => (
                      <span key={topicIndex} className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-primary-100 text-primary-700">
                        {topic}
                      </span>
                    ))}
                    {(module.keyTopics || []).length > 3 && (
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-700">
                        +{(module.keyTopics || []).length - 3} more
                      </span>
                    )}
                  </div>
                </div>
                <ChevronRight className="h-6 w-6 text-gray-400" />
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Mind Map Modal */}
      {mindMapData && (
        <MindMap 
          data={mindMapData} 
          onClose={() => setMindMapData(null)} 
        />
      )}
    </div>
  );
}

function GenerateCoursePage() {
  const [formData, setFormData] = useState({
    topic: '',
    difficulty: 'intermediate',
  });
  const [generatedCourse, setGeneratedCourse] = useState<Course | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);
    setError(null);
    try {
      const response = await axios.post('http://127.0.0.1:8001/generate-course/', {
        topic: formData.topic,
        difficulty: formData.difficulty,
      });
      
      const courseData = typeof response.data === 'string' 
        ? JSON.parse(response.data) 
        : response.data;
      
      if (!courseData.courseTitle || !courseData.courseOverview || !courseData.modules) {
        console.error("Missing required properties in course data");
        setError('Course data is missing required properties.');
        return;
      }
      
      const formattedCourse = {
        courseTitle: courseData.courseTitle,
        courseOverview: courseData.courseOverview,
        modules: courseData.modules || []
      };

      // Create course history entry
      const courseHistoryEntry: CourseHistory = {
        ...formattedCourse,
        id: crypto.randomUUID(),
        createdAt: new Date().toISOString(),
      };

      // Get existing course history
      const existingHistory = JSON.parse(sessionStorage.getItem('courseHistory') || '[]');
      
      // Add new course to history
      sessionStorage.setItem('courseHistory', JSON.stringify([courseHistoryEntry, ...existingHistory]));

      setGeneratedCourse(formattedCourse);
    } catch (error) {
      console.error('Error generating course:', error);
      setError('Failed to generate course content. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  if (generatedCourse) {
    return <CourseDisplay course={generatedCourse} onBack={() => setGeneratedCourse(null)} />;
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-lg shadow px-6 py-8">
          <div className="mb-8 text-center">
            <Sparkles className="mx-auto h-12 w-12 text-primary-600" />
            <h2 className="mt-4 text-3xl font-bold text-gray-900">Generate New Course ✨</h2>
            <p className="mt-2 text-gray-600">Fill in the details to generate your custom course</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="topic" className="block text-sm font-medium text-gray-700">
                Course Topic 📚
              </label>
              <input
                type="text"
                id="topic"
                value={formData.topic}
                onChange={(e) => setFormData({ ...formData, topic: e.target.value })}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                placeholder="e.g., Machine Learning Basics"
                required
              />
            </div>

            <div>
              <label htmlFor="difficulty" className="block text-sm font-medium text-gray-700">
                Difficulty Level 📊
              </label>
              <select
                id="difficulty"
                value={formData.difficulty}
                onChange={(e) => setFormData({ ...formData, difficulty: e.target.value })}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={isGenerating}
              className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
                isGenerating ? 'bg-primary-400 cursor-not-allowed' : 'bg-primary-600 hover:bg-primary-700'
              } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500`}
            >
              {isGenerating ? 'Generating Course...' : 'Generate Course 🚀'}
            </button>
          </form>

          {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
        </div>
      </div>
    </div>
  );
}

function CoursesPage() {
  const [courseHistory, setCourseHistory] = useState<CourseHistory[]>([]);
  const [selectedCourse, setSelectedCourse] = useState<CourseHistory | null>(null);

  useEffect(() => {
    const history = JSON.parse(sessionStorage.getItem('courseHistory') || '[]');
    setCourseHistory(history);
  }, []);

  if (selectedCourse) {
    return <CourseDisplay course={selectedCourse} onBack={() => setSelectedCourse(null)} />;
  }

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="border-b border-gray-200 px-6 py-4">
        <h2 className="text-xl font-semibold text-gray-900">Generated Courses History 📚</h2>
      </div>

      <div className="p-6">
        {courseHistory.length === 0 ? (
          <div className="text-center py-12">
            <BookText className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No courses yet</h3>
            <p className="mt-1 text-sm text-gray-500">Get started by generating your first course</p>
          </div>
        ) : (
          <div className="space-y-4">
            {courseHistory.map((course) => (
              <button
                key={course.id}
                onClick={() => setSelectedCourse(course)}
                className="w-full text-left border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow bg-white"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900">{course.courseTitle}</h3>
                    <p className="text-sm text-gray-500 mt-1">
                      Created {new Date(course.createdAt).toLocaleDateString()} at{' '}
                      {new Date(course.createdAt).toLocaleTimeString()}
                    </p>
                    <p className="text-gray-600 mt-2">{course.courseOverview}</p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {course.modules.slice(0, 3).map((module, index) => (
                        <span
                          key={index}
                          className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-primary-100 text-primary-700"
                        >
                          {module.moduleTitle}
                        </span>
                      ))}
                      {course.modules.length > 3 && (
                        <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-700">
                          +{course.modules.length - 3} more modules
                        </span>
                      )}
                    </div>
                  </div>
                  <ChevronRight className="h-6 w-6 text-gray-400" />
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [selectedTab, setSelectedTab] = useState('courses');

  if (!isLoggedIn) {
    return <LoginPage onLogin={() => setIsLoggedIn(true)} />;
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <BookOpen className="h-8 w-8 text-primary-600" />
              <span className="ml-2 text-xl font-bold text-gray-900">Craftify</span>
            </div>
            <div className="flex items-center space-x-4">
              <img
                className="h-8 w-8 rounded-full"
                src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?fit=facearea&facepad=2&w=256&h=256&q=80"
                alt="User avatar"
              />
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="col-span-1">
            <div className="bg-white rounded-lg shadow p-4">
              <div className="space-y-2">
                <button
                  onClick={() => setSelectedTab('courses')}
                  className={`w-full text-left px-4 py-2 rounded-md flex items-center ${
                    selectedTab === 'courses' ? 'bg-primary-50 text-primary-600' : 'hover:bg-gray-50'
                  }`}
                >
                  <BookOpen className="h-5 w-5 mr-2" />
                  Courses 📚
                </button>
                <button
                  onClick={() => setSelectedTab('generate')}
                  className={`w-full text-left px-4 py-2 rounded-md flex items-center ${
                    selectedTab === 'generate' ? 'bg-primary-50 text-primary-600' : 'hover:bg-gray-50'
                  }`}
                >
                  <Sparkles className="h-5 w-5 mr-2" />
                  Generate ✨
                </button>
              </div>
            </div>
          </div>

          {/* Main Content Area */}
          <div className="col-span-1 md:col-span-3">
            {selectedTab === 'generate' ? (
              <GenerateCoursePage />
            ) : (
              <CoursesPage />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;