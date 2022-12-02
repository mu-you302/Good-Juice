import json
import socket
from base import *
from req import *
from resp import *
from config import config
from ui import UI
import subprocess
import logging
import re
from threading import Thread
from itertools import cycle
from time import sleep
import sys
from collections import namedtuple
import random
import numpy as np

Point = namedtuple("Point", ["x", "y"])

# logger config
logging.basicConfig(
    # uncomment this will redirect log to file *client.log*
    # filename="client.log",
    format="[%(asctime)s][%(levelname)s] %(message)s",
    filemode="a+",
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# fp = open("./resp.json", "w+")

# record the context of global data
gContext = {
    "playerID": None,
    "characterID": [],
    "gameOverFlag": False,
    "prompt": (
        "Take actions!\n"
        "'s': move in current direction\n"
        "'w': turn up\n"
        "'e': turn up right\n"
        "'d': turn down right\n"
        "'x': turn down\n"
        "'z': turn down left\n"
        "'a': turn up left\n"
        "'u': sneak\n"
        "'i': unsneak\n"
        "'j': master weapon attack\n"
        "'k': slave weapon attack\n"
        "Please complete all actions within one frame! \n"
        "[example]: a12sdq2\n"
    ),
    "steps": ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"],
    "gameBeginFlag": False,
}


class Client(object):
    """Client obj that send/recv packet.

    Usage:
        >>> with Client() as client: # create a socket according to config file
        >>>     client.connect()     # connect to remote
        >>>
    """

    def __init__(self) -> None:
        self.config = config
        self.host = self.config.get("Host")
        self.port = self.config.get("Port")
        assert self.host and self.port, "host and port must be provided"
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        if self.socket.connect_ex((self.host, self.port)) == 0:
            logger.info(f"connect to {self.host}:{self.port}")
        else:
            logger.error(f"can not connect to {self.host}:{self.port}")
            exit(-1)
        return

    def send(self, req: PacketReq):
        msg = json.dumps(req, cls=JsonEncoder).encode("utf-8")
        length = len(msg)
        self.socket.sendall(length.to_bytes(8, sys.byteorder) + msg)
        return

    def recv(self):
        length = int.from_bytes(self.socket.recv(8), sys.byteorder)
        result = b""
        while resp := self.socket.recv(length):
            result += resp
            length -= len(resp)
            if length <= 0:
                break

        # global fp

        jsonRes = json.loads(result)
        # json.dump(jsonRes, fp)

        packet = PacketResp().from_json(result)
        return packet, jsonRes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.socket.close()
        if traceback:
            print(traceback)
            return False
        return True


def kmeans(data, k=3, normalize=False, limit=500):
    """基于numpy实现kmeans聚类"""
    if normalize:
        stats = (data.mean(axis=0), data.std(axis=0))
        data = (data - stats[0]) / stats[1]

    np.random.shuffle(data)
    centers = data[:k]

    for i in range(limit):
        classifications = np.argmin(
            ((data[:, :, None] - centers.T[None, :, :]) ** 2).sum(axis=1), axis=1
        )
        new_centers = np.array(
            [data[classifications == j, :].mean(axis=0) for j in range(k)]
        )

        if (new_centers == centers).all():
            break
        else:
            centers = new_centers

    if normalize:
        centers = centers * stats[1] + stats[0]

    return classifications, centers


class Model(object):
    direc2action = {0: "w", 1: "e", 2: "d", 3: "x", 4: "z", 5: "a"}
    iterNum = 0
    InitCall = True  # 区分是否为第一次调用

    def __init__(self, attr: dict) -> None:
        # 其中坐标的存储方式为[[x, y]...]
        self.mapinfo = {
            "notmine": [],  # 对方占领的或者空白的
            "buff_move": [],
            "buff_hp": [],
        }
        self.obstacles = [False] * (16 * 16)  # 地图中的障碍是不变的
        self.enemyinfo = None
        self.color = None
        self.attr = attr
        self.desitination = None

    @staticmethod
    def Dist(P1: Point, P2: Point) -> int:
        """计算两点之间的距离"""
        return abs(P1.x - P2.x) + abs(P1.y - P2.y)

    @staticmethod
    def MovePos(point: Point, direction: int) -> Point:
        """返回向一个方向移动之后的坐标变化

        Args:
            Point (_type_): _description_
            direction (_type_): 0-5
        """
        if direction == 0:
            return Point(point.x - 1, point.y + 1)
        elif direction == 1:
            return Point(point.x - 1, point.y)
        elif direction == 2:
            return Point(point.x, point.y - 1)
        elif direction == 3:
            return Point(point.x + 1, point.y - 1)
        elif direction == 4:
            return Point(point.x + 1, point.y)
        else:
            return Point(point.x, point.y + 1)

    @staticmethod
    def AvoidSlave(direc: int) -> int:
        """躲避路径上的 SlaveWeapon

        Args:
            direc (int): 0-5
        """
        direcList = list(range(6))
        direcList.remove(direcList)
        direcList.remove((direc + 3) % 6)
        return random.choice(direcList)

    @staticmethod
    def IsOut(point: Point) -> bool:
        """判断坐标是否超出界限"""
        return point.x < 0 or point.x > 15 or point.y > 0 or point.y < -15

    @staticmethod
    def axis2idx(x, y) -> int:
        """根据坐标返回在数组中的索引
        地图中的格子排列:x=0, y=0~-15,x=1...所以[x, y] 在数组中对应的下标为 x*16-y
        """
        return 16 * x - y

    def ExistSlave(self, Pos: Point, blocks: list[dict]) -> bool:
        """判断以Pos为中心的一圈中是否存在SlaveWeapon

        Args:
            Pos (Point): 中心
            blocks (list[dict]): _description_

        Returns:
            bool: _description_
        """
        P_all = [Pos]
        for i in range(6):
            P_all.append(Model.MovePos(Pos, i))

        if sum([self.HasSlave(P, blocks) for P in P_all]) > 0:
            return True

        return False

    def HasSlave(self, Pos: Point, blocks) -> bool:
        """判断一个格子中是否有SlaveWeapon

        Args:
            Pos (Point): _description_
            block (_type_): _description_

        Returns:
            _type_: _description_
        """
        if Model.IsOut(Pos):
            return False
        idx = Model.axis2idx(*Pos)
        block = blocks[idx]
        if "objs" in block:
            obj = block["objs"][-1]["status"]
            if "weaponType" in obj and obj["playerID"] != self.attr["playerID"]:
                return True

        return False

    def ObstacleNums(self, Pos: list[int]) -> int:
        """计算一个点周围的障碍总数

        Args:
            Pos (list[int]): _description_

        Returns:
            int: _description_
        """
        Pos = Point(Pos[0], Pos[1])
        P_all = [Pos]
        for i in range(6):
            P_n = Model.MovePos(Pos, i)
            if Model.IsOut(P_n):
                continue
            P_all.append(P_n)

        idxs = [Model.axis2idx(*P) for P in P_all]
        return sum([self.obstacles[i] for i in idxs])

    def action(self, jsonCon: dict) -> str:
        """返回动作序列

        Args:
            jsonCon (dict): 接收到的内容

        Returns:
            str: _description_
        """

        res = ""
        if jsonCon["type"] == PacketType.GameOver:
            return ""
        attri = jsonCon["data"]["characters"][0]
        if self.color is None:
            self.color = attri["color"]

        MoveAble = attri["moveCDLeft"] == 0  # 判断是否能移动
        MasterAble = attri["masterWeapon"]["attackCDLeft"] == 0
        SlaveAble = attri["slaveWeapon"]["attackCDLeft"] == 0

        # 获取自己当前的位置
        myPositionX = attri["x"]
        myPositionY = attri["y"]
        myPosition = Point(myPositionX, myPositionY)

        blocks = jsonCon["data"]["map"]["blocks"]

        # 确定移动方向
        if MoveAble:
            direc = None
            # 判断周围是否有武器和buff
            for k in range(6):
                newPos = Model.MovePos(myPosition, k)
                if self.IsOut(newPos):
                    continue
                block = blocks[Model.axis2idx(newPos.x, newPos.y)]
                if "objs" in block:
                    obj = block["objs"][0]
                    if (
                        "weaponType" in obj
                        and obj["status"]["playerID"] != self.attr["playerID"]
                    ):
                        direc = Model.AvoidSlave(k)
                        break
                    if "buffType" in obj["status"]:
                        direc = k
                        break

            # 避开无敌状态的敌方
            if self.enemyinfo["isGod"]:
                enemyPos = Point(self.enemyinfo["x"], self.enemyinfo["y"])
                if Model.Dist(enemyPos, myPosition) <= 4:
                    direc = Model.ToDirection(enemyPos, myPosition)

            if direc is None:
                self.GetDestination(myPosition, jsonCon["data"]["frame"])
                direc = Model.ToDirection(myPosition, self.desitination)
                x, y = Model.MovePos(myPosition, direc)
                # 如果撞墙，不太好处理，先简单反向
                if (
                    Model.IsOut(Point(x, y))
                    or not blocks[Model.axis2idx(x, y)]["valid"]
                ):
                    direc = (direc + 3) % 6

            # 是否潜行
            nextPos = Model.MovePos(myPosition, direc)
            if blocks[Model.axis2idx(*nextPos)]["color"] == self.color:
                res += Model.direc2action[direc] + "su"
            else:
                res += Model.direc2action[direc] + "is"

        if MasterAble:
            direc = self.DurianDirec(myPosition, blocks)
            res += Model.direc2action[direc] + "ij"

        if SlaveAble:
            direc = self.KiwiDirec(myPosition, blocks)
            res += Model.direc2action[direc] + "ik"

        Model.iterNum += 1
        if Model.iterNum == 6:
            Model.iterNum = 0
            self.iterMap(blocks)  # 每6 frame进行一次全地图迭代
        return res

    def DurianDirec(self, myPos: Point, blocks: list[dict]) -> int:
        """返回释放猕猴桃的最佳方向"""
        # 得到6个方向的中心坐标增量
        incre = [[-2, 2], [-2, 0], [0, -2], [2, -2], [2, 0], [0, 2]]
        Add = lambda P1, P: Point(P1.x + P[0], P1.y + P[1])
        EnemyPos = Point(self.enemyinfo["x"], self.enemyinfo["x"])
        ContainEnemy = lambda P1, P: True if P1.x == P.x and P1.y == P.y else False
        num = [0] * 6
        for i in range(6):
            P_center = Add(myPos, incre[i])
            P_all = [P_center]  # 一个方向上覆盖的所有点
            for j in range(6):
                P_all.append(Model.MovePos(P_center, j))
            if sum([ContainEnemy(Pos, EnemyPos) for Pos in P_all]) > 0:
                return i
            # 如果范围内有敌方武器
            if self.ExistSlave(P_center, blocks):
                return i
            num[i] = sum([self.IsOccupiable(Pos, blocks) for Pos in P_all])

        return num.index(max(num))

    def KiwiDirec(self, myPos: Point, blocks: list[dict]) -> int:
        """返回释放猕猴桃的最佳方向"""
        incre = [[-1, 1], [-1, 0], [0, -1], [1, -1], [1, 0], [0, 1]]
        Add = lambda P1, P: Point(P1.x + P[0], P1.y + P[1])
        num = [0] * 6
        for i in range(6):
            Pos = myPos  # 初始位置
            for j in range(8):
                Pos = Add(Pos, incre[i])
                if self.IsOccupiable(Pos, blocks):
                    num[i] += 1

        return num.index(max(num))

    def IsOccupiable(self, Pos, blocks) -> bool:
        """判断一个格子是否能被占据"""
        if Model.IsOut(Pos):
            return False
        idx = Model.axis2idx(*Pos)
        block = blocks[idx]
        return (not self.obstacles[idx]) and block["color"] != self.color

    def iterMap(self, blocks: list[dict], init: bool = False):
        """遍历地图中的blocks数组获取信息

        Args:
            blocks (list[dict]): length:16*16
            call:self.iterMap(jsonCon["data"]["map"]["blocks"])
            init: if it's the fitst call
        """
        # 先将已有的信息清空
        for v in self.mapinfo.values():
            v.clear()

        for block in blocks:
            if not block["valid"]:
                if Model.InitCall:
                    idx = Model.axis2idx(block["x"], block["y"])
                    self.obstacles[idx] = True
                else:
                    continue
            if block["color"] != self.color:
                self.mapinfo["notmine"].append([block["x"], block["y"]])
            if "objs" in block:  # 有特殊单位
                obj = block["objs"][-1]
                if (
                    obj["type"] == 1
                    and obj["status"]["playerID"] != self.attr["playerID"]
                ):
                    self.enemyinfo = obj["status"]  # 记录敌方信息
                elif obj["type"] == 2 and obj["status"]["buffType"] == 1:
                    self.mapinfo["buff_move"].append([block["x"], block["y"]])
                elif obj["type"] == 2 and obj["status"]["buffType"] == 2:
                    self.mapinfo["buff_hp"].append([block["x"], block["y"]])

        if init:
            Model.InitCall = False

        return

    def GetDestination(self, myPos: Point, round: int) -> None:
        """计算下一个要前往的目标点

        Args:
            myPos (Point): _description_
            round (int): 回合数

        Returns:
            _type_: None
        """
        if self.desitination is not None:
            if Model.Dist(myPos, self.desitination) >= 3:
                return  # 还没有到达上一个目标点，不更新

        # 如果要挂了，就获取生命buff
        if self.attr["hp"] < 50:
            dist = []
            for Pos in self.mapinfo["buff_hp"]:
                dist.append(Model.Dist(Point(*Pos), myPos))
            if dist and min(dist) < 6:
                idx = dist.index(min(dist))
                self.desitination = Point(*self.mapinfo["buff_hp"][idx])
                return

        # 判断附近是否有移速buff
        if self.attr["moveCD"] > 1:
            dist = []
            for Pos in self.mapinfo["buff_move"]:
                dist.append(Model.Dist(Point(*Pos), myPos))
            if dist and min(dist) < 6:
                idx = dist.index(min(dist))
                self.desitination = Point(*self.mapinfo["buff_move"][idx])
                return

        X = np.array(self.mapinfo["notmine"])
        X_mine = np.array(myPos)
        n_cluster = min(int(round / 60) + 1, 5)

        labels, centers = kmeans(X, normalize=True, k=n_cluster)
        # 统计每一个cluster中的点数量
        num_clusters = np.bincount(labels)
        # 计算每个聚类中心离当前位置的距离
        dist2center = np.abs(centers - X_mine).sum(axis=1)
        # 综合考虑
        num_obstacles = []
        for i in centers:
            num_obstacles.append(self.ObstacleNums(i.astype(np.int32)))

        # 计算目标点周围的障碍数，防止进圈卡住
        Onum = (
            np.array(num_obstacles) if round > 600 / 3 else np.zeros(dist2center.shape)
        )
        cret = -0.6 * num_clusters + dist2center + 0.6 * Onum
        # cret = -0.6 * num_clusters + dist2center
        maxIdx = np.argmin(cret)
        self.desitination = Point(*centers[maxIdx])
        return

    def ToDirection(P1: Point, P2: Point) -> int:
        """返回从P1移动到P2的大致方向"""
        if P2.x >= P1.x and P2.y >= P1.y:
            return random.choice([4, 5])
        elif P2.x >= P1.x and P2.y <= P1.y:
            return random.choice([2, 3, 4])
        elif P2.x <= P1.x and P2.y <= P1.y:
            return random.choice([1, 2])
        else:
            return random.choice([0, 1, 5])


def cliGetInitReq():
    """Get init request from user input."""
    # masterWeaponType = input("Make choices!\nmaster weapon type: [select from {1-2}]: ")
    masterWeaponType = 2
    # slaveWeaponType = input("slave weapon type: [select from {1-2}]: ")
    slaveWeaponType = 1
    return InitReq(
        MasterWeaponType(int(masterWeaponType)), SlaveWeaponType(int(slaveWeaponType))
    )


def cliGetActionReq(characterID: int, actions: str):
    """Get action request according to actions.

    Args:
        characterID (int): Character's id that do actions.
    """

    def get_action(s: str):
        regex = r"[swedxzauijk]"
        matches = re.finditer(regex, s)
        for match in matches:
            yield match.group()

    str2action = {
        "s": (ActionType.Move, EmptyActionParam()),
        "w": (ActionType.TurnAround, TurnAroundActionParam(Direction.Above)),
        "e": (ActionType.TurnAround, TurnAroundActionParam(Direction.TopRight)),
        "d": (ActionType.TurnAround, TurnAroundActionParam(Direction.BottomRight)),
        "x": (ActionType.TurnAround, TurnAroundActionParam(Direction.Bottom)),
        "z": (ActionType.TurnAround, TurnAroundActionParam(Direction.BottomLeft)),
        "a": (ActionType.TurnAround, TurnAroundActionParam(Direction.TopLeft)),
        "u": (ActionType.Sneaky, EmptyActionParam()),
        "i": (ActionType.UnSneaky, EmptyActionParam()),
        "j": (ActionType.MasterWeaponAttack, EmptyActionParam()),
        "k": (ActionType.SlaveWeaponAttack, EmptyActionParam()),
    }

    actionReqs = []

    # actions = input()

    for s in get_action(actions):
        actionReq = ActionReq(characterID, *str2action[s])
        actionReqs.append(actionReq)

    return actionReqs


def refreshUI(ui: UI, packet: PacketResp):
    """Refresh the UI according to the response."""
    data = packet.data
    if packet.type == PacketType.ActionResp:
        ui.playerID = data.playerID
        ui.color = data.color
        ui.characters = data.characters
        ui.score = data.score
        ui.kill = data.kill

        for block in data.map.blocks:
            if len(block.objs):
                ui.block = {
                    "x": block.x,
                    "y": block.y,
                    "color": block.color,
                    "valid": block.valid,
                    "obj": block.objs[-1].type,
                    "data": block.objs[-1].status,
                }
            else:
                ui.block = {
                    "x": block.x,
                    "y": block.y,
                    "color": block.color,
                    "valid": block.valid,
                    "obj": ObjType.Null,
                }
    subprocess.run(["clear"])
    ui.display()


def recvAndRefresh(ui: UI, client: Client):
    """Recv packet and refresh ui."""
    global gContext
    resp, res = client.recv()
    refreshUI(ui, resp)

    if resp.type == PacketType.ActionResp:
        if len(resp.data.characters) and not gContext["gameBeginFlag"]:
            gContext["characterID"] = resp.data.characters[-1].characterID
            gContext["playerID"] = resp.data.playerID
            gContext["gameBeginFlag"] = True

            model = Model(res["data"]["characters"][-1])
            model.iterMap(res["data"]["map"]["blocks"], init=True)  # 遍历map

    while resp.type != PacketType.GameOver:
        resp, res = client.recv()

        # 模型决策
        actionStr = model.action(res)
        action = cliGetActionReq(gContext["characterID"], actionStr)
        actionPacket = PacketReq(PacketType.ActionReq, action)
        client.send(actionPacket)

        refreshUI(ui, resp)

    refreshUI(ui, resp)
    print(f"Game Over!")

    for (idx, score) in enumerate(resp.data.scores):
        if gContext["playerID"] == idx:
            print(f"You've got \33[1m{score} score\33[0m")
        else:
            print(f"The other player has got \33[1m{score} score \33[0m")

    if resp.data.result == ResultType.Win:
        print("\33[1mCongratulations! You win! \33[0m")
    elif resp.data.result == ResultType.Tie:
        print("\33[1mEvenly matched opponent \33[0m")
    elif resp.data.result == ResultType.Lose:
        print(
            "\33[1mThe goddess of victory is not on your side this time, but there is still a chance next time!\33[0m"
        )

    gContext["gameOverFlag"] = True
    print("Press any key to exit......")


def main():
    ui = UI()

    with Client() as client:
        client.connect()

        initPacket = PacketReq(PacketType.InitReq, cliGetInitReq())
        client.send(initPacket)
        print(gContext["prompt"])

        # IO thread to display UI
        t = Thread(target=recvAndRefresh, args=(ui, client))
        t.start()

        for c in cycle(gContext["steps"]):
            if gContext["gameBeginFlag"]:
                break
            print(
                f"\r\033[0;32m{c}\033[0m \33[1mWaiting for the other player to connect...\033[0m",
                flush=True,
                end="",
            )
            sleep(0.1)

        # IO thread accepts user input and sends requests
        # while not gContext["gameOverFlag"]:
        #     if gContext["characterID"] is None:
        #         continue
        #     if action := cliGetActionReq(gContext["characterID"]):
        #         actionPacket = PacketReq(PacketType.ActionReq, action)
        #         client.send(actionPacket)

        # gracefully shutdown
        t.join()


if __name__ == "__main__":
    main()
