from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ShopifyStores(BaseModel):
    pass

class User(BaseModel):
    id: str
    firstName: str
    lastName: str
    email: str
    industry: Optional[str] = None
    firebaseUid: str
    role: str
    ownerId: Optional[str] = None
    expertLanguages: List[str] = Field(default_factory=list)
    expertiseLevel: str
    invitedBy: Optional[str] = None
    invitationStatus: str
    shopifyState: Optional[str] = None
    stripeCustomerId: Optional[str] = None
    subscriptionPlan: str
    subscriptionStatus: str
    wordsUsedThisMonth: int
    wordsLimit: int
    isActive: bool
    lastLoginAt: Optional[datetime] = None
    shopifyStores: List[ShopifyStores] = Field(default_factory=list)
    createdAt: datetime
    updatedAt: datetime
    v: int = Field(alias="__v")
