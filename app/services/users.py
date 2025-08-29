from ..mongodb import users_collection


def get_user_profile(user_id: str):
    return users_collection.find_one({"_id": user_id})


# def get_user_glossary(user_id: str):
#     return glossary_collection.find_one({"user_id": user_id})
