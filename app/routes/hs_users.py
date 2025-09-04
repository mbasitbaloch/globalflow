from fastapi import APIRouter, Depends, HTTPException, status
from ..mongoSc import User
from typing import List
from ..mongodb import users_collection
from bson import ObjectId

router = APIRouter(
    prefix="/user",
    tags=["Users"]
)

def user_helper(user) -> dict:
    return {
        "id" : str(user["_id"]),
        "firstName" : user.get("firstName"),
        "lastName" : user.get("lastName"),
        "email" : user.get("email"),
        "industry" : user.get("industry"),
        "firebaseUid" : user.get("firebaseUid"),
        "role" : user.get("role"),
        "ownerId" : user.get("ownerId"),
        "expertLanguages" : user.get("expertLanguages", []),
        "expertiseLevel" : user.get("expertiseLevel"),
        "invitedBy" : user.get("invitedBy"),
        "invitationStatus" : user.get("invitationStatus"),
        "shopifyState" : user.get("shopifyState"),
        "stripeCustomerId" : user.get("stripeCustomerId"),
        "subscriptionPlan" : user.get("subscriptionPlan"),
        "subscriptionStatus" : user.get("subscriptionStatus"),
        "wordsUsedThisMonth" : user.get("wordsUsedThisMonth"),
        "wordsLimit" : user.get("wordsLimit"),
        "isActive" : user.get("isActive"),
        "lastLoginAt" : user.get("lastLoginAt"),
        "shopifyStores" : user.get("shopifyStores", []),
        "createdAt" : user.get("createdAt"),
        "updatedAt" : user.get("updatedAt"),
        "__v" : user.get("__v")
    }

@router.get("/get_all", response_model=List[User], status_code=status.HTTP_200_OK)
async def get_all():
    users = users_collection.find({})
    return [user_helper(user) for user in users]

@router.get("/{id}", response_model=User, status_code=status.HTTP_200_OK)
async def get_user(user_id):
    user = users_collection.find_one({"_id":ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found!")
    return user_helper(user)