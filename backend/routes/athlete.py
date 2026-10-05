from fastapi import APIRouter, Depends

from core.deps import require_role
from routes.activity_catalog import ACTIVITY_IDS_BY_ROLE


router = APIRouter(
    prefix="/api/athlete",
    tags=["athlete"],
    dependencies=[Depends(require_role("athlete"))],
)


@router.get("/activities")
async def get_activities() -> dict[str, list[str]]:
    return {"activities": list(ACTIVITY_IDS_BY_ROLE["athlete"])}
