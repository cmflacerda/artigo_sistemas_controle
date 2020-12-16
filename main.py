#!/usr/bin/env python3
from mujoco_py import load_model_from_xml, MjSim, MjViewer,load_model_from_path
import numpy as np
import math
from numpy import pi

# model = load_model_from_path("./assets/two_link.xml")
print("Escolha o corpo que sera submetido a analise: ")
print("1 - corpo rigido")
print("2 - corpo macio")
menu = int(input("Escolha: "))

while menu!=1 and menu!=2:
    print("Valor incorreto")
    print("1 - corpo rigido")
    print("2 - corpo macio")
    menu = int(input("Escolha: "))

if menu == 1:
    model = load_model_from_path("./assets/full_kuka_two_joints.xml")
elif menu == 2:
    model = load_model_from_path("./assets/full_kuka_two_joints_soft_body.xml")

file = open("dados_forca.txt", "a")


sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
l1 = sim.model.geom_size[1]
l2 = sim.model.geom_size[2]

erro_anterior = 0
tempo = 0
erro_integracao = 0
theta_virtual_antigo = 0

m1 = 7
m2 = 1.8
g = 9.81
L1 = 0.210
L2 = 0.063
Izz1 = (1/12)*m1*(L1**2)
Izz2 = (1/12)*m2*(L2**2)

passo_theta = (1*pi)/180


# ai is the distance from Zi-1 to Zi along Xi-1
# alphai is the angle from Zi-1 to Zi about Xi-1
# di is the distance from Xi-1 to Xi along Zi
# thetai is the angle from Xi-1 to Xi about Zi

# Tabela Denavit-Hartenberg utilizada no desenvolvimento do codigo
# Link | ai | alphai | di | thetai
#   1  |  0 |   0    | 0  | theta1
#   2  |2L1 |   pi   | 0  | theta2 - pi/2
#   3  |  0 |   pi/2 | 2L2|   0

# os comprimentos L1 e L2, de acordo com modelagem dinamica, são as
# distancias do eixo de rotacao ate o centro de massa de cada link

# Matriz Denavit-Hartenberg
# |  cos(thetai)             -sin(thetai)             0             ai             |
# |  sin(thetai)cos(alphai)  cos(thetai)cos(alphai)  -sin(alphai)  -sin(alphai)di  |
# |  sin(thetai)sin(alphai)  cos(thetai)sin(alphai)   cos(alphai)   cos(alphai)di  |
# |  0                       0                        0             1              |

# Matrizes de transformacao
# 0T1 = |  cos(theta1)             -sin(theta1)             0             0             |
#       |  sin(theta1)              cos(theta1)             0             0             |
#       |  0                        0                       1             0             |
#       |  0                        0                       0             1             |

# 1T2 = |  cos(theta2 - pi/2)      -sin(theta2 - pi/2)      0             2L1           |
#       | -sin(theta2 - pi/2)      -cos(theta2 - pi/2)      0             0             |
#       |  0                        0                      -1             0             |
#       |  0                        0                       0             1             |

# 2T3 = |  1                        0                       0             0             |
#       |  0                        0                      -1            -2L2           |
#       |  0                        1                       0             0             |
#       |  0                        0                       0             1             |

# 0T2 = |  cos(theta1+theta2 - pi/2)      -sin(theta1+theta2 - pi/2)             0             cos(theta1)2L1  |
#       |  sin(theta1-theta2 + pi/2)      -cos(theta1-theta2 + pi/2)             0             sin(theta1)2L1  |
#       |  0                        0                             -1             0               |
#       |  0                        0                              0             1               |

# 0T3 = |  cos(theta1+theta2 - pi/2)       0                              sin(theta1+theta2 - pi/2) cos(theta1)2L1 + sin(theta1+theta2 - pi/2)2L2|
#       |  sin(theta1-theta2 + pi/2)       0                              cos(theta1+theta2 - pi/2) sin(theta1)2L1 + cos(theta1-theta2 + pi/2)2L2|
#       |  0                       -1                              0             0                                     |
#       |  0                        0                              0             1                                     |

# Construcao da Jacobiana
# a = cos(theta1)2L1 + sin(theta1+theta2 - pi/2)2L2
# b = sin(theta1)2L1 + cos(theta1-theta2 + pi/2)2L2
# w_theta1_base = [0, 0, 1]^t
# w_theta2_base = [0, 0, 1]^t

# J = |  da/dtheta1  da/dtheta2  |
#     |  db/dtheta1  db/dtheta2  |
#     |  0           0           |
#     |  0           0           |
#     |  1           1           |

def fowardkin(theta1, theta2):

    # 0T3
    matriz_transformacao = [[np.cos(theta1+theta2 - pi/2), 0, np.sin(theta1+theta2 - pi/2), (np.cos(theta1)*2*L1 + np.sin(theta1+theta2 - pi/2)*2*L2)],
          [np.sin(theta1-theta2 + pi/2), 0, np.cos(theta1+theta2 - pi/2), (np.sin(theta1)*2*L1 + np.cos(theta1-theta2 + pi/2)*2*L2)],
          [0, -1, 0, 0],
          [0, 0, 0, 1]]

    # r_ee
    posicao_ee = [[0], [0], [0], [1]]

    # r_base = 0T3 * r_ee
    posicao_base = np.matmul(matriz_transformacao, posicao_ee)
    posicao_base_final = [[posicao_base[0,0]], [posicao_base[1,0]], [posicao_base[2,0]]]

    return posicao_base_final

def jacobian(theta1, theta2):

    matriz_jacobiana = np.array([[2*(L2*np.sin(theta1 + theta2) - L1*np.sin(theta1)), 2*L2*np.sin(theta1 + theta2)],
                        [-2*(L2*np.cos(theta1 - theta2) - L1*np.cos(theta1)), 2*L2*np.cos(theta2 - theta1)],
                        [0, 0]])
                        #[0, 0],
                        #[1, 1]])

    return matriz_jacobiana

def control_action(theta1, theta2, theta_virtual):

    posicao_angular_atual = [[sim.data.qpos[0]], [sim.data.qpos[1]]]
    velocidade_angular_atual = np.array([sim.data.qvel[0], sim.data.qvel[1]])


    # matrizes dinamica
    Mq = [[(m1*L1*L1 + Izz1 + 4*L1*L1*m2*Izz2), 2*m2*L1*L2*(math.sin(theta1)*math.sin(theta2) + math.cos(theta1)*math.cos(theta2))],
          [2*m2*L1*L2*(math.sin(theta1)*math.sin(theta2) + math.cos(theta1)*math.cos(theta2)), (m2*L2*L2 + Izz2)]]

    Gq = [[(m1*L1 + 2*m2*L1)*g],[m2*L2*math.cos(theta2)*g]]

    Sq = [[2*m2*L1*L2*(math.sin(theta1)*math.cos(theta2) - math.cos(theta1)*math.sin(theta2))*velocidade_angular_atual[1]**2],
          [2*m2*L1*L2*(math.cos(theta1)*math.sin(theta2) - math.sin(theta1)*math.cos(theta2))*velocidade_angular_atual[0]**2]]

    kp = 350
    kv = 50

    Kp = kp * np.eye(3) #np.array([[kp, 0, 0],[],[]])
    Kv = kv * np.eye(3)

    if menu == 1:
        kj_junta_um = 100
        kj_junta_dois = 40
        bj_junta_um = 5
        bj_junta_dois = 2

    elif menu == 2:
        kj_junta_um = 3.2
        kj_junta_dois = 5
        bj_junta_um = 2
        bj_junta_dois = 2

    Kj = [[kj_junta_um, 0], [0, kj_junta_dois]]
    Bj = [[bj_junta_um, 0], [0, bj_junta_dois]]

    posicao_virtual = fowardkin(1.570796327, 0) # 1.570796327 theta_virtual

    if menu == 1:
        if theta_virtual < 2.4:
            posicao_virtual_juntas = [[-theta_virtual], [1.570796327]]
        elif theta_virtual >= 2.4:
            posicao_virtual_juntas = [[-2.4], [1.570796327]]
    elif menu == 2:
        if posicao_angular_atual[1][0] < 1.570796327:
            posicao_virtual_juntas = [[0], [theta_virtual]]
        elif posicao_angular_atual[1][0] >= 1.570796327:
            posicao_virtual_juntas = [[-1.570796327], [1.570796327]]

    #print(posicao_virtual)
    velocidade_virtual = [[0], [0], [0]]
    velocidade_virtual_juntas = [[0], [0]]

    posicao_angular_atual = [[sim.data.qpos[0]],[sim.data.qpos[1]]]
    velocidade_angular_atual = [[sim.data.qvel[0]], [sim.data.qvel[1]]]

    theta1 = posicao_angular_atual[0][0]
    theta2 = posicao_angular_atual[1][0]
    posicao_atual = fowardkin(theta1, theta2)

    J = jacobian(theta1,theta2)

    delta_posicao = np.subtract(posicao_virtual, posicao_atual)
    delta_posicao_junta = np.subtract(posicao_virtual_juntas, posicao_angular_atual)
    delta_velocidade = np.subtract(velocidade_virtual, np.matmul(J, velocidade_angular_atual))
    delta_velocidade_junta = np.subtract(velocidade_virtual_juntas, velocidade_angular_atual)


    T_a = np.matmul(J.transpose(), np.matmul(Kp, delta_posicao) + np.matmul(Kv, delta_velocidade))
    T_a_juntas = np.matmul(Kj, delta_posicao_junta) + np.matmul(Bj, delta_velocidade_junta)
    #print(T_a)
    #sim.data.ctrl[0]= T_a[0][0]
    #sim.data.ctrl[1] = T_a[1][0]

    sim.data.ctrl[0] = T_a_juntas[0][0]
    sim.data.ctrl[1] = T_a_juntas[1][0]

    print(sim.data.ctrl)
    print("")

    #print(f'{sim.data.sensordata:.1f}')
    print("coordenada x global da forca: " + str(sim.data.sensordata[2]))
    print("coordenada y global da forca: " + str(sim.data.sensordata[0]))
    print("coordenada z global da forca: " + str(sim.data.sensordata[1]))
    print("")
    print("coordenada x global do torque: " + str(sim.data.sensordata[5]))
    print("coordenada y global do torque: " + str(sim.data.sensordata[3]))
    print("coordenada z global do torque: " + str(sim.data.sensordata[4]))
    print("")

    modulo_forca = np.sqrt(sim.data.sensordata[2]**2 + sim.data.sensordata[0]**2 + sim.data.sensordata[1]**2)
    file.write(str(round(sim.data.sensordata[2],2)) + "  " + str(round(sim.data.sensordata[0],2)) + "  " + str(round(sim.data.sensordata[1],2)) + "  " + str(round(modulo_forca,2)) + "  " + str(sim.data.time) + "\n")


try:
    while True:
        viewer.render()
        sim.step()

        #print("ee xpos = ", sim.data.site_xpos[0])
        q = sim.data.qpos

        posicao_cartesiana = np.array([sim.data.body_xpos[0], sim.data.body_xpos[1]])
        posicao_angular_atual =np.array([sim.data.qpos[0],sim.data.qpos[1]])
        velocidade_angular_atual =np.array([sim.data.qvel[0],sim.data.qvel[1]])
        #print(q)
        theta_virtual = passo_theta + theta_virtual_antigo
        theta_virtual_antigo = theta_virtual

        control_action(sim.data.qpos[0],sim.data.qpos[1], theta_virtual)
        #print(fowardkin(0,0))
        #qd = np.array([np.pi/3,0])
        #vd = np.array([0,0])

        #erro = [qd[0]-posicao_angular_atual[0],qd[1]-posicao_angular_atual[1]]
        #erro = qd - posicao_angular_atual
        #erro_um = vd - velocidade_angular_atual
        #print(erro)


        #delta_t = sim.data.time - tempo
        #media_erro = ((erro + erro_anterior)/2)*delta_t # integracao do erro
        #erro_integracao += media_erro
        #erro_anterior = erro
        #tempo = sim.data.time

        #kp = 10
        #kv = 2
        #ki = 10


        #Kp = kp*np.eye(2)
        #Kv = kv*np.eye(2)
        #Ki = ki*np.eye(2)


        #control_action = np.dot(Kp,erro)
        #control_action = np.dot(Kp, erro) + np.dot(Kv, erro_um)
        #control_action = np.dot(Kp, erro) + np.dot(Kv, erro_um) + np.dot(Ki, erro_integracao)

        #sim.data.ctrl[0]= control_action[0]
        #sim.data.ctrl[1] = control_action[1]
        #print(u)

        #print(sim.model.body_mass)


        #print(sim.data.body_xpos[0, 2])
        #print(sim.data.body_xpos[1, 2])

        #print(posicao_angular_atual)
        #print(posicao_cartesiana)
        #x = [l1*np.cos(q[0])+l2*np.cos(q[0]+q[1]), l1*np.sin(q[0])+l2*np.sin(q[0]+q[1])]
        #print("forward kin = ", x) # sim.data.site_jacp[0].reshape((3,-1))

        t += 1
        # if t > 500:
        #     break

except KeyboardInterrupt:
    print("saindo")