#!/usr/bin/env python3
from mujoco_py import load_model_from_xml, MjSim, MjViewer,load_model_from_path
import numpy as np

#model = load_model_from_path("./assets/two_link.xml")
model = load_model_from_path("./assets/full_kuka_two_joints.xml")
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0

L1 = sim.model.geom_size[1]
L2 = sim.model.geom_size[2]



erro_anterior = 0
tempo = 0
erro_integracao = 0
flag = 0
posicao_angular_atual =np.array([sim.data.qpos[0],sim.data.qpos[1]])
qd = np.array([np.pi/3,posicao_angular_atual[0]])
erro = qd - posicao_angular_atual
try:
    while True:
        #viewer.render()
        #sim.step()

        #print("ee xpos = ", sim.data.site_xpos[0])
        #q = sim.data.qpos

        #posicao_angular_atual =np.array([sim.data.qpos[0],sim.data.qpos[1]])
        #velocidade_angular_atual =np.array([sim.data.qvel[0],sim.data.qvel[1]])
        #print(q)

        #qd = np.array([np.pi/3,posicao_angular_atual[0]])
        #vd = np.array([0,0])

        #erro = [qd[0]-posicao_angular_atual[0],qd[1]-posicao_angular_atual[1]]
        #erro = qd - posicao_angular_atual
        #if erro[0] <= 0.11:
         #   flag = 1
        #erro_um = vd - velocidade_angular_atual
        #print(erro)

        if erro[0] <= 0.12 and flag == 1:

            viewer.render()
            sim.step()

            # print("ee xpos = ", sim.data.site_xpos[0])
            q = sim.data.qpos

            posicao_angular_atual = np.array([sim.data.qpos[0], sim.data.qpos[1]])
            velocidade_angular_atual = np.array([sim.data.qvel[0], sim.data.qvel[1]])
            # print(q)

            # qd = np.array([np.pi/3,posicao_angular_atual[0]])
            vd = np.array([0, 0])

            # erro = [qd[0]-posicao_angular_atual[0],qd[1]-posicao_angular_atual[1]]
            erro = qd - posicao_angular_atual
            if erro[0] <= 0.12:
                flag = 1
            erro_um = vd - velocidade_angular_atual
            # print(erro)

            qd = np.array([np.pi / -3, posicao_angular_atual[0]])
            erro = qd - posicao_angular_atual

            delta_t = sim.data.time - tempo
            media_erro = ((erro + erro_anterior)/2)*delta_t # integracao do erro
            erro_integracao += media_erro
            erro_anterior = erro
            tempo = sim.data.time

            kp = 10
            kv = 2
            ki = 10


            Kp = kp*np.eye(2)
            Kv = kv*np.eye(2)
            Ki = ki*np.eye(2)


            #control_action = np.dot(Kp,erro)
            control_action = np.dot(Kp, erro) + np.dot(Kv, erro_um)
            #control_action = np.dot(Kp, erro) + np.dot(Kv, erro_um) + np.dot(Ki, erro_integracao)

            sim.data.ctrl[0]= control_action[0]
            sim.data.ctrl[1] = control_action[1]
            #print(u)

            #print(sim.model.body_mass)




            print(posicao_angular_atual)
            x = [L1*np.cos(q[0])+L2*np.cos(q[0]+q[1]), L1*np.sin(q[0])+L2*np.sin(q[0]+q[1])]
            #print("forward kin = ", x) # sim.data.site_jacp[0].reshape((3,-1))

            #print(erro)
            t += 1
            # if t > 500:
            #     break

            if posicao_angular_atual[0] <= -0.9:
                flag = 0
                erro[0] = erro[0] * (-1)
                print(erro)

        elif erro[0] > 0.12 and flag == 0:

            viewer.render()
            sim.step()

            # print("ee xpos = ", sim.data.site_xpos[0])
            q = sim.data.qpos

            posicao_angular_atual = np.array([sim.data.qpos[0], sim.data.qpos[1]])
            velocidade_angular_atual = np.array([sim.data.qvel[0], sim.data.qvel[1]])
            # print(q)

            qd = np.array([np.pi/3,posicao_angular_atual[0]])
            vd = np.array([0, 0])

            # erro = [qd[0]-posicao_angular_atual[0],qd[1]-posicao_angular_atual[1]]
            erro = qd - posicao_angular_atual
            if erro[0] <= 0.12:
                flag = 1
            erro_um = vd - velocidade_angular_atual
            # print(erro)

            delta_t = sim.data.time - tempo
            media_erro = ((erro + erro_anterior) / 2) * delta_t  # integracao do erro
            erro_integracao += media_erro
            erro_anterior = erro
            tempo = sim.data.time

            kp = 10
            kv = 2
            ki = 10

            Kp = kp * np.eye(2)
            Kv = kv * np.eye(2)
            Ki = ki * np.eye(2)

            # control_action = np.dot(Kp,erro)
            control_action = np.dot(Kp, erro) + np.dot(Kv, erro_um)
            # control_action = np.dot(Kp, erro) + np.dot(Kv, erro_um) + np.dot(Ki, erro_integracao)

            sim.data.ctrl[0] = control_action[0]
            sim.data.ctrl[1] = control_action[1]
            # print(u)

            # print(sim.model.body_mass)

            print(posicao_angular_atual)
            x = [L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]), L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])]
            # print("forward kin = ", x) # sim.data.site_jacp[0].reshape((3,-1))

            #print(erro)
            t += 1
            # if t > 500:
            #     break

except KeyboardInterrupt:
    print("saindo")