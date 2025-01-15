from tool.run_HGR import solve_question_from_logic_forms
from utils.pymysql_comm import UsingMysql

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from passlib.context import CryptContext
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import JWTError, jwt
from http import HTTPStatus
from starlette.responses import JSONResponse

SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def get_user_db():
    with UsingMysql(log_time=False) as um:
        um.cursor.execute('select username,hashed_password,disabled from user')
        data_list = um.cursor.fetchall()

    users_db = {}

    for row in data_list:
        name = row['username']
        info = dict(username=name, hashed_password=row['hashed_password'], disabled=row['disabled'])
        dic = {name: info}
        users_db.update(dic)

    return users_db


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(
    title="FASTAPI service",
    desription="Intelligent tutoring system API",
)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(get_user_db(), username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(get_user_db(), form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/", tags=["General"])
def read_root(current_user: User = Depends(get_current_active_user)):
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "user": current_user.username,
    }
    return response


class APGD(BaseModel):
    point_instances: List[str] = Field(..., description="List of point names, e.g. ['A','B','C']")
    line_instances: List[str] = Field(..., description="List of line names, e.g. ['AB','AC']")
    circle_instances: List[str] = Field(..., description="List of circle names, can be empty")
    point_positions: Dict[str, List[float]] = Field(..., description="Mapping point -> [x,y]")
    logic_forms: List[str] = Field(..., description="List of logic forms, e.g. ['Equals(LengthOf(Line(A,B)), y)']")


@app.post("/solve_apgd", tags=["APGD"])
def solve_apgd(apgd: APGD):
    result = solve_question_from_logic_forms(apgd.dict())
    return JSONResponse({
        'solution': result
    })
