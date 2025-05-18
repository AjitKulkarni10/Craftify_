from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from course_generator import take_user_input_and_create_course, generate_mindmap_data, take_user_input_and_create_test
from youtube import search_youtube, get_video_url , get_transcript
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CourseRequest(BaseModel):
    topic: str
    difficulty: str

class QuizRequest(BaseModel):
    topic: str

class VideoRequest(BaseModel):
    topic: str

@app.post("/generate-course/")
async def generate_course(request: CourseRequest):
    try:
        print(f"Generating course for topic: {request.topic} with difficulty: {request.difficulty}")
        
        course_content = take_user_input_and_create_course(request.topic, request.difficulty)
        print("Course content received:", course_content)
        
        if isinstance(course_content, str):
            import json
            try:
                # This will raise an error if the JSON is invalid
                course_content = json.loads(course_content)
                print("Successfully parsed JSON string")
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print("Raw content:", course_content)
                raise HTTPException(
                    status_code=500, 
                    detail=f"Invalid JSON generated: {str(e)}"
                )
        
        # Validate required fields
        required_fields = ["courseTitle", "courseOverview", "modules"]
        missing_fields = [field for field in required_fields if field not in course_content]
        if missing_fields:
            print(f"Missing required fields: {missing_fields}")
            raise HTTPException(
                status_code=500,
                detail=f"Generated course is missing required fields: {', '.join(missing_fields)}"
            )
        
        # Validate modules
        if not isinstance(course_content["modules"], list):
            print("Modules is not a list")
            raise HTTPException(
                status_code=500,
                detail="Generated course modules must be a list"
            )
        
        if len(course_content["modules"]) == 0:
            print("No modules generated")
            raise HTTPException(
                status_code=500,
                detail="Generated course must have at least one module"
            )
        
        # Validate each module
        for i, module in enumerate(course_content["modules"]):
            if not isinstance(module, dict):
                print(f"Module {i} is not a dictionary")
                raise HTTPException(
                    status_code=500,
                    detail=f"Module {i + 1} is not properly formatted"
                )
            
            required_module_fields = ["moduleTitle", "moduleOverview", "keyTopics", "detailedContent"]
            missing_module_fields = [field for field in required_module_fields if field not in module]
            if missing_module_fields:
                print(f"Module {i} missing fields: {missing_module_fields}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Module {i + 1} is missing required fields: {', '.join(missing_module_fields)}"
                )
        
        return course_content
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        print("Full traceback:", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate course: {str(e)}"
        )


@app.post("/generate-mindmap/")
async def generate_mindmap(course_content: dict):
    try:
        mindmap_data = generate_mindmap_data(course_content)
        if not mindmap_data:
            raise HTTPException(status_code=500, detail="Failed to generate mindmap data")
        return mindmap_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-quiz/")
async def generate_quiz(request: QuizRequest):
    try:
        quiz_content = take_user_input_and_create_test(request.topic)        
        return quiz_content
    
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate quiz: {str(e)}"
        )

@app.post("/get-video/")
async def get_video(request: VideoRequest):
    try:
        video_id = search_youtube(request.topic)
        if not video_id:
            raise HTTPException(status_code=404, detail="No video found for the given topic")
        
        video_url = get_video_url(video_id)
        transcript = get_transcript(video_id)
        return {"video_url": video_url, "video_id": video_id , "transcript" : transcript}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get video: {str(e)}"
        )
    

@app.get("/")
def read_root():
    return {"message": "Backend is running"}
