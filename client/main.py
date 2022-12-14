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
        # uncomment this will show req packet
        # logger.info(f"send PacketReq, content: {msg}")
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
        # new_centers = np.array(
        #     [data[classifications == j, :].mean(axis=0) for j in range(k)]
        # )
        # 可能有一类一个样本也没有分到
        new_centers = np.array(
            [
                data[classifications == j, :].mean(axis=0)
                for j in np.unique(classifications)
            ]
        )
        if new_centers.shape[0] != centers.shape[0]:
            continue

        if (new_centers == centers).all():
            break
        else:
            centers = new_centers

    if normalize:
        centers = centers * stats[1] + stats[0]

    return classifications, centers


class Model(object):
    direc2action = {0: "w", 1: "e", 2: "d", 3: "x", 4: "z", 5: "a"}
    blockAxis = np.array(
        [[x, y] for x in range(24) for y in reversed(range(-23, 1))]
    )  # 全部的格点
    # 生成障碍位置
    obstacleIdx = [
        21,
        22,
        23,
        45,
        46,
        47,
        69,
        70,
        71,
        93,
        94,
        95,
        117,
        118,
        119,
        141,
        142,
        143,
        165,
        166,
        167,
        189,
        190,
        191,
        213,
        214,
        215,
        237,
        238,
        239,
        261,
        262,
        263,
        285,
        286,
        287,
        309,
        310,
        311,
        333,
        334,
        335,
        357,
        358,
        359,
        381,
        382,
        383,
        405,
        406,
        407,
        429,
        430,
        431,
        453,
        454,
        455,
        477,
        478,
        479,
        501,
        502,
        503,
        504,
        505,
        506,
        507,
        508,
        509,
        510,
        511,
        512,
        513,
        514,
        515,
        516,
        517,
        518,
        519,
        520,
        521,
        522,
        523,
        524,
        525,
        526,
        527,
        528,
        529,
        530,
        531,
        532,
        533,
        534,
        535,
        536,
        537,
        538,
        539,
        540,
        541,
        542,
        543,
        544,
        545,
        546,
        547,
        548,
        549,
        550,
        551,
        552,
        553,
        554,
        555,
        556,
        557,
        558,
        559,
        560,
        561,
        562,
        563,
        564,
        565,
        566,
        567,
        568,
        569,
        570,
        571,
        572,
        573,
        574,
        575,
    ]
    iterNum = 0

    def __init__(self, chars: list[dict]) -> None:
        # 其中坐标的存储方式为[[x, y], ...]
        self.mapinfo = {
            "notmine": [True] * (24 * 24),  # True表示不是自己的，可以配合blockAxis转换成坐标
            "buff_move": [],
            "buff_hp": [],
        }
        self.enemyinfo = None
        self.color = None
        self.chars = chars
        self.desitination = [None, None]

    @staticmethod
    def Dist(P1: Point, P2: Point) -> int:
        """计算两点之间的距离"""
        return (
            abs(P1.x - P2.x) + abs(P1.y - P2.y) + abs(P1.x + P1.y - P2.x - P2.y)
        ) / 2

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
        direcList.remove(direc)
        direcList.remove((direc + 3) % 6)
        return random.choice(direcList)

    @staticmethod
    def IsOut(point: Point) -> bool:
        """判断坐标是否超出界限"""
        return point.x < 0 or point.x > 23 or point.y > 0 or point.y < -23

    @staticmethod
    def axis2idx(x, y) -> int:
        """根据坐标返回在数组中的索引
        地图中的格子排列:x=0, y=0~-23,x=1...所以[x, y] 在数组中对应的下标为 x*24-y
        """
        return 24 * x - y

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

    def action_old(self, jsonCon: dict) -> str:
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
            self.iterMap(blocks)
        return res

    def action(self, jsonCon: dict) -> str:
        """返回两个角色应该采取的动作"""
        self.chars = jsonCon["data"]["characters"]
        if self.color is None:
            self.color = self.chars[0]["color"]

        actionList = []

        self.iterMap(jsonCon["data"]["map"]["blocks"])

        for i in range(2):
            res = ""
            attri = self.chars[i]
            MoveAble = attri["moveCDLeft"] == 0  # 判断是否能移动
            MasterAble = attri["masterWeapon"]["attackCDLeft"] == 0
            SlaveAble = attri["slaveWeapon"]["attackCDLeft"] == 0

            myPos = Point(attri["x"], attri["y"])

            if MoveAble:
                direc = None
                # 判断周围是否有buff
                for k in range(6):
                    newPos = Model.MovePos(myPos, k)
                    if self.IsOut(newPos):
                        continue
                    if [newPos.x, newPos.y] in self.mapinfo["buff_move"] or [
                        newPos.x,
                        newPos.y,
                    ] in self.mapinfo["buff_hp"]:
                        direc = k
                        break

                if direc is None:
                    self.GetDestination(i, myPos, jsonCon["data"]["frame"])
                    direc = Model.ToDirection(myPos, self.desitination[i])
                    x, y = Model.MovePos(myPos, direc)
                    # 如果撞墙，不太好处理，先简单反向
                    if Model.IsOut(Point(x, y)):
                        direc = (direc + 3) % 6

                res += Model.direc2action[direc] + "s"

            if MasterAble:
                direc = self.DurianDirec(myPos)
                res += Model.direc2action[direc] + "j"
            if SlaveAble:
                direc = self.KiwiDirec(myPos)
                res += Model.direc2action[direc] + "k"
            direc = self.GetDirection(myPos, jsonCon["data"]["map"]["blocks"])
            res += Model.direc2action[direc]

            actionList.append(res)

        return actionList

    def DurianDirec(self, myPos: Point) -> int:
        """返回释放榴莲的最佳方向"""
        # 得到6个方向的中心坐标增量
        incre = [[-2, 2], [-2, 0], [0, -2], [2, -2], [2, 0], [0, 2]]
        Add = lambda P1, P: Point(P1.x + P[0], P1.y + P[1])
        EnemyPos = (
            Point(self.enemyinfo["x"], self.enemyinfo["y"]) if self.enemyinfo else None
        )
        ContainEnemy = lambda P1, P: True if P1.x == P.x and P1.y == P.y else False
        num = [0] * 6
        for i in range(6):
            P_center = Add(myPos, incre[i])
            P_all = [P_center]  # 一个方向上覆盖的所有点
            for j in range(6):
                P_all.append(Model.MovePos(P_center, j))
            # if EnemyPos and sum([ContainEnemy(Pos, EnemyPos) for Pos in P_all]) > 0:
            #     return i
            # 如果范围内有敌方武器
            # if self.ExistSlave(P_center, blocks):
            #     return i
            num[i] = sum([self.IsOccupiable(Pos) for Pos in P_all])

        return num.index(max(num))

    def KiwiDirec(self, myPos: Point) -> int:
        """返回释放猕猴桃的最佳方向"""
        incre = [[-1, 1], [-1, 0], [0, -1], [1, -1], [1, 0], [0, 1]]
        Add = lambda P1, P: Point(P1.x + P[0], P1.y + P[1])
        num = [0] * 6
        for i in range(6):
            Pos = myPos  # 初始位置
            for j in range(8):
                Pos = Add(Pos, incre[i])
                if self.IsOccupiable(Pos):
                    num[i] += 1

        return num.index(max(num))

    def CactusDirec(self, myPos: Point) -> int:
        """返回释放仙人掌的最佳方向

        Args:
            myPos (Point): 当前位置

        Returns:
            int: 方向，0-5
        """
        pass

    def IsOccupiable(self, Pos) -> bool:
        """判断一个格子是否能被占据"""
        if Model.IsOut(Pos):
            return False
        idx = Model.axis2idx(*Pos)
        return self.mapinfo["notmine"][idx]

    def iterMap(self, blocks: list[dict]):
        """遍历地图中的blocks数组获取信息
        Args:
            blocks (list[dict]): length:16*16
            call:self.iterMap(jsonCon["data"]["map"]["blocks"])
            init: if it's the fitst call
        """
        # 每过6轮将已有的信息清空
        if Model.iterNum == 6:
            for v in self.mapinfo.values():
                v.clear()

            Model.iterNum = 0
        else:
            Model.iterNum += 1

        self.mapinfo["notmine"] = [True] * (24 * 24)
        for i in Model.obstacleIdx:
            self.mapinfo["notmine"][i] = False

        for block in blocks:
            if not block["valid"]:
                continue
            if block["color"] == self.color:  # 标记自己的格点
                x, y = block["x"], block["y"]
                self.mapinfo["notmine"][Model.axis2idx(x, y)] = False
            if "objs" in block and block["x"] < 21 and block["y"] > -21:  # 有不在障碍区内的特殊单位
                obj = block["objs"][-1]
                if (
                    obj["type"] == 1
                    and obj["status"]["playerID"] != self.chars[0]["playerID"]
                ):
                    self.enemyinfo = obj["status"]  # 记录敌方信息
                elif obj["type"] == 2 and obj["status"]["buffType"] == 1:
                    self.mapinfo["buff_move"].append([block["x"], block["y"]])
                elif obj["type"] == 2 and obj["status"]["buffType"] == 2:
                    self.mapinfo["buff_hp"].append([block["x"], block["y"]])

        return

    def GetDestination(self, order: int, myPos: Point, round) -> None:
        """获取目标点

        Args:
            order (_type_): character序号, 0/1
            myPos (_type_): _description_
            round (_type_): _description_

        Returns:
            _type_: 更新self.destination
        """
        if self.desitination[order] is not None:
            if Model.Dist(myPos, self.desitination[order]) >= 3:
                return  # 还没有到达上一个目标点，不更新

        if self.chars[order]["hp"] < 50:
            dist = []
            for Pos in self.mapinfo["buff_hp"]:
                dist.append(Model.Dist(Point(*Pos), myPos))
            if dist and min(dist) < 6:
                idx = dist.index(min(dist))
                self.desitination[order] = Point(*self.mapinfo["buff_hp"][idx])
                # self.mapinfo["buff_hp"].pop(idx)
                return

        # 判断附近是否有移速buff
        if self.chars[order]["moveCD"] > 1:
            dist = []
            for Pos in self.mapinfo["buff_move"]:
                dist.append(Model.Dist(Point(*Pos), myPos))
            if dist and min(dist) < 6:
                idx = dist.index(min(dist))
                self.desitination[order] = Point(*self.mapinfo["buff_move"][idx])
                return

        X = Model.blockAxis[self.mapinfo["notmine"]]
        X_mine = np.array(myPos)
        n_cluster = min(int(round / 60) + 1, 5)

        labels, centers = kmeans(X, k=n_cluster)
        # 统计每一个cluster中的点数量
        num_clusters = np.bincount(labels)
        # 直接用bincount得到的结果可能少了一个
        if num_clusters.shape[0] != centers.shape[0]:
            num_clusters = np.append(num_clusters, 0)
        # 计算每个聚类中心离当前位置的距离
        dist2center = np.abs(centers - X_mine).sum(axis=1)
        # 综合考虑
        cret = -0.6 * num_clusters + dist2center

        maxIdx = np.argmin(cret)

        des = centers[maxIdx]
        des[0] = min(des[0], 20)
        des[1] = max(des[1], -20)
        self.desitination[order] = Point(*des)

    def GetDirection(self, myPos: Point, blocks: list[dict]) -> int:
        """返回回合结束时向哪个方向获取的信息最多

        Args:
            myPos (Point): _description_

        Returns:
            int: 0-5
        """
        num = [0] * 12
        for block in blocks:
            if not block["valid"]:
                continue
            if block["color"] == self.color:
                x, y = block["x"], block["y"]
                if Model.Dist(Point(x, y), myPos) <= 5:
                    if x < myPos.x and y == myPos.y:
                        num[0] += 1
                    elif x == myPos.x and y < myPos.y:
                        num[2] += 1
                    elif x > myPos.x and x + y == myPos.x + myPos.y:
                        num[4] += 1
                    elif x > myPos.x and y == myPos.y:
                        num[6] += 1
                    elif x == myPos.x and y > myPos.y:
                        num[8] += 1
                    elif x + y == myPos.x + myPos.y and y > myPos.y:
                        num[10] += 1
                    elif x < myPos.x and y < myPos.y:
                        num[1] += 1
                    elif x > myPos.x and y > myPos.y:
                        num[7] += 1
                    elif x > myPos.x and x + y < myPos.x + myPos.y:
                        num[3] += 1
                    elif x > myPos.x and x + y > myPos.x + myPos.y:
                        num[5] += 1
                    elif x < myPos.y and x + y > myPos.x + myPos.y:
                        num[9] += 1
                    elif x < myPos.y and x + y < myPos.x + myPos.y:
                        num[11] += 1
        Num = [0] * 6
        for i in range(6):
            for j in range(5):
                Num[i] += num[(8 + i * 2 + j) % 12]
        # 这里可以引入随机变量，选择最少的三个中的一个方向，我不会python...
        n = random.choice(sorted(Num)[:3])
        return Num.index(n)

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


def cliGetInitReq(master: int, slave: int):
    """Get init request from user input."""
    masterWeaponType = str(master)  # 西瓜为1，榴莲为2
    slaveWeaponType = str(slave)  # 猕猴桃1，仙人掌2
    return InitReq(
        MasterWeaponType(int(masterWeaponType)), SlaveWeaponType(int(slaveWeaponType))
    )


def cliGetActionReq(characterID: int, actions):
    """Get action request from user input.

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
        ui.frame = data.frame

        for block in data.map.blocks:
            if len(block.objs):
                ui.block = {
                    "x": block.x,
                    "y": block.y,
                    "color": block.color,
                    "valid": block.valid,
                    "frame": block.frame,
                    "obj": block.objs[-1].type,
                    "data": block.objs[-1].status,
                }
            else:
                ui.block = {
                    "x": block.x,
                    "y": block.y,
                    "color": block.color,
                    "valid": block.valid,
                    "frame": block.frame,
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
            gContext["characterID"] = [
                character.characterID for character in resp.data.characters
            ]
            gContext["playerID"] = resp.data.playerID
            gContext["gameBeginFlag"] = True

            model = Model(res["data"]["characters"])
            # model.iterMap(res["data"]["map"]["blocks"], init=True)  # 遍历map

    while resp.type != PacketType.GameOver:
        resp, res = client.recv()
        if res["type"] == PacketType.GameOver:
            break

        refreshUI(ui, resp)

        actionStr = model.action(res)
        # 两个角色分别发送
        for i in range(2):
            action = cliGetActionReq(gContext["characterID"][i], actionStr[i])
            actionPacket = PacketReq(PacketType.ActionReq, action)
            client.send(actionPacket)

    refreshUI(ui, resp)
    print(f"Game Over!")

    for (idx, score) in enumerate(resp.data.scores):
        if gContext["playerID"] == idx:
            print(f"You've got {score} score.")
        else:
            print(f"The other player has got {score} score.")

    if resp.data.result == ResultType.Win:
        print("Congratulations! You win! ")
    elif resp.data.result == ResultType.Tie:
        print("Evenly matched opponent ")
    elif resp.data.result == ResultType.Lose:
        print("@@@@@@@@ You lost! ")

    gContext["gameOverFlag"] = True
    print("Press any key to exit......")
    print("=" * 30 + "\n")


def main():
    ui = UI()

    with Client() as client:
        client.connect()

        initPacket = PacketReq(
            PacketType.InitReq, [cliGetInitReq(2, 1), cliGetInitReq(2, 1)]
        )
        client.send(initPacket)
        # print(gContext["prompt"])

        # IO thread to display UI
        # t = Thread(target=recvAndRefresh, args=(ui, client))
        # t.start()

        # for c in cycle(gContext["steps"]):
        #     if gContext["gameBeginFlag"]:
        #         break
        #     print(
        #         " Waiting for the other player to connect...",
        #         flush=True,
        #         end="",
        #     )
        #     sleep(0.1)

        # IO thread accepts user input and sends requests
        # while gContext["gameOverFlag"] is False:
        #     if not gContext["characterID"]:
        #         continue
        #     for characterID in gContext["characterID"]:
        #         if action := cliGetActionReq(characterID):
        #             actionPacket = PacketReq(PacketType.ActionReq, action)
        #             client.send(actionPacket)

        # gracefully shutdown
        # t.join()

        recvAndRefresh(ui, client)


if __name__ == "__main__":
    main()
