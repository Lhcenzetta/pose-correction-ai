from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from schema.Exerciceschema import ExerciseCreate, ExerciseBase, ExerciseResponse
from routers.authentication import get_current_user, verfiy_token
from db.database import get_db
from schema import UserSchema
from models.exercices import Exercice

router = APIRouter()


@router.post("/", response_model=ExerciseResponse)
def create_exercise(
    exercie: ExerciseCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    existing_exercise = db.query(Exercice).filter(Exercice.name == exercie.name).first()
    if existing_exercise:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An exercise with this name already exists.",
        )
    if exercie.name == "Shoulder Abduction":
        description = (
            "The elbow is a hinge joint in the upper limb formed by "
            "the articulation of three bones: the humerus (upper arm bone), "
            "radius (forearm bone on the thumb side), and ulna (forearm bone on the little finger side).  "
            "It allows flexion and extension of the arm, enabling movements like bending and straightening"
        )
    elif exercie.name == "arm abduction":
        description = (
            "Arm abduction is the movement of the arm away from the body’s "
            "midline in the coronal (frontal) plane.  It begins with the arm at the side and "
            "progresses upward, such as when raising the arm to the side or overhead. This motion is "
            "initiated by the supraspinatus (first 15°), continued by the deltoid (up to 90°), and completed by "
            "upward rotation of the scapula driven by the trapezius and serratus anterior beyond 90°.  Full abduction can reach 160–180° "
            "and is essential for activities like performing a jumping jack."
        )
    else:
        description = "Dosen't exit now , maybe later"

    new_exerice = Exercice(
        name=exercie.name, description=description, duration_time=exercie.duration_time
    )
    db.add(new_exerice)
    db.commit()
    db.refresh(new_exerice)
    return new_exerice


@router.get("/exercises", response_model=list[ExerciseResponse])
def get_all_exercises(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    exercises = db.query(Exercice).all()
    return exercises
