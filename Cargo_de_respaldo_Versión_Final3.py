import pandapower as pp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.utils import resample

# Definir número de iteraciones
num_iteraciones = 400

diferencias = []
Pbase = []

# Iterar n veces el análisis
for iteracion in range(num_iteraciones):
    
    print(f"--------- Iteración {iteracion + 1} / {num_iteraciones} ------------")

    # CREACIÓN DE LA RED IEEE 39 NODOS
    net = pp.create_empty_network()

    # CREACIÓN DE BARRAS
    df = pd.read_excel(open("IEEE39NODES.xlsx", 'rb'), sheet_name="BUS")
    for idx in df.index:
        pp.create_bus(net, name=df.at[idx, "name"], vn_kv=df.at[idx, "vn_kv"],
                      max_vm_pu=df.at[idx, "max_vm_pu"], min_vm_pu=df.at[idx, "min_vm_pu"])

    # CREACIÓN DE LÍNEAS
    df = pd.read_excel(open("IEEE39NODES.xlsx", 'rb'), sheet_name="LINE")
    for idx in df.index:
        pp.create_line_from_parameters(net, name=df.at[idx, "name"], from_bus=df.at[idx, "from_bus"],
                                       to_bus=df.at[idx, "to_bus"], length_km=df.at[idx, "length_km"],
                                       r_ohm_per_km=df.at[idx, "r_ohm_per_km"], x_ohm_per_km=df.at[idx, "x_ohm_per_km"],
                                       c_nf_per_km=df.at[idx, "c_nf_per_km"], max_i_ka=df.at[idx, "max_i_ka"])

    # CREACIÓN DE CARGAS
    df = pd.read_excel(open("IEEE39NODES.xlsx", 'rb'), sheet_name="LOAD")
    for idx in df.index:
        pp.create_load(net, name=df.at[idx, "name"], bus=df.at[idx, "bus"],
                       p_mw=df.at[idx, "p_mw"], q_mvar=df.at[idx, "q_mvar"])
        Datos_load = pd.DataFrame(net.load)

    # CREACIÓN DE TRANSFORMADORES
    df = pd.read_excel(open("IEEE39NODES.xlsx", 'rb'), sheet_name="TRAFO")
    for idx in df.index:
        pp.create_transformer_from_parameters(net, name=df.at[idx, "name"], hv_bus=df.at[idx, "hv_bus"],
                                              lv_bus=df.at[idx, "lv_bus"], sn_mva=df.at[idx, "sn_mva"],
                                              vn_hv_kv=df.at[idx, "vn_hv_kv"], vn_lv_kv=df.at[idx, "vn_lv_kv"],
                                              vk_percent=df.at[idx, "vk_percent"], vkr_percent=df.at[idx, "vkr_percent"],
                                              pfe_kw=df.at[idx, "pfe_kw"], i0_percent=df.at[idx, "i0_percent"])

    # CREACIÓN DE GENERADORES
    df = pd.read_excel(open("IEEE39NODES.xlsx", 'rb'), sheet_name="GEN")
    for idx in df.index:
        pp.create_gen(net, name=df.at[idx, "name"], bus=df.at[idx, "bus"],
                      p_mw=df.at[idx, "p_mw"], vm_pu=df.at[idx, "vm_pu"], slack=df.at[idx, "slack"])

    # CREACIÓN DE SLACK
    df = pd.read_excel(open("IEEE39NODES.xlsx", 'rb'), sheet_name="SLACK")
    for idx in df.index:
        pp.create_ext_grid(net, bus=df.at[idx, "bus"], vm_pu=df.at[idx, "vm_pu"], va_degree=df.at[idx, "va_degree"])
        
    # LISTAS PARA ALMACENAR LOS DATOS DEL NODO 30
    potencia_nodo30_sin_gd = []
    potencia_nodo30_con_gd = []
    potencia_otros_nodos_sin_gd = []
    potencia_otros_nodos_con_gd = []
    
    # CURVAS DE CARGA
    curva_carga_ind = pd.DataFrame()
    curva_carga_res = pd.DataFrame()
    curva_carga_com = pd.DataFrame()

    Nodos_industriales = [0,1,3,6,8,11,14,18]
    Nodos_residenciales = [2,4,7,10,12,13,15,17]
    Nodos_comerciales = [5,9,16]
     
    # ===============================
    # 🔹 CREACIÓN DE CURVAS DE CARGA
    # ===============================

    for ind in Nodos_industriales:
        carga_hora_ind = []  # Inicializar lista para este nodo

        for i in range(24):  
            if 6 < i < 22:    
                u = np.random.uniform(1.2, 1.5)          
            else:
                u = np.random.uniform(0.7, 0.9)
            
            carga_ajustada_ind = net.load.iloc[ind]['p_mw'] * u   
            carga_hora_ind.append(carga_ajustada_ind)  # Guardar todas las horas
        
        curva_carga_ind[ind] = carga_hora_ind  # Guardar nodo en DataFrame

    for res in Nodos_residenciales:
        carga_hora_res = []  # Inicializar lista para este nodo

        for i in range(24):  
            if 6 < i < 8:    
                u = np.random.uniform(1.2, 1.4)           
            elif 7 < i < 17:
                u = np.random.uniform(0.6, 0.7)
            elif 16 < i < 20:
                u = np.random.uniform(0.8, 1.0)
            else:
                u = np.random.uniform(0.6, 0.7)
            
            carga_ajustada_res = net.load.iloc[res]['p_mw'] * u  
            carga_hora_res.append(carga_ajustada_res)  # Guardar todas las horas
        
        curva_carga_res[res] = carga_hora_res  # Guardar nodo en DataFrame     

    for com in Nodos_comerciales:
        carga_hora_com = []  # Inicializar lista para este nodo

        for i in range(24):  
            if 9 < i < 21:    
                u = np.random.uniform(1.2, 1.4)           
            else:
                u = np.random.uniform(0.5, 0.7)
            
            carga_ajustada_com = net.load.iloc[com]['p_mw'] * u  
            carga_hora_com.append(carga_ajustada_com)  # Guardar todas las horas
            
        curva_carga_com[com] = carga_hora_com  # Guardar nodo en DataFrame

    plt.figure(figsize=(12, 6))

    # Graficar cada nodo industrial
    for nodo in curva_carga_ind.columns:
        plt.plot(curva_carga_ind.index, curva_carga_ind[nodo], linestyle="-", marker="o", label=f"Nodo Industrial {nodo}")

    # Graficar cada nodo residencial
    for nodo in curva_carga_res.columns:
        plt.plot(curva_carga_res.index, curva_carga_res[nodo], linestyle="-", marker="s", label=f"Nodo Residencial {nodo}")

    # Graficar cada nodo comercial
    for nodo in curva_carga_com.columns:
        plt.plot(curva_carga_com.index, curva_carga_com[nodo], linestyle="-", marker="v", label=f"Nodo comercial {nodo}")

    # Configuración de la gráfica
    plt.xlabel("Horas")
    plt.ylabel("Demanda (MW)")
    plt.title("Curva de Carga de 24 Horas para Cada Nodo")
    plt.grid(True)
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))  # Mueve la leyenda a la derecha
    # Guardar la gráfica en un archivo
    grafico_archivo = f"Curva_de_demanda_residencial.png"
    plt.savefig(grafico_archivo)
    plt.close()  # Cerrar la gráfica para que no se muestre

    # ===============================
    # 🔹 ANALISIS SIN GD
    # ===============================

    print("--------- Analisis sin GD ------------")

    for h in range (24):
        
        print("Hora de analisis: ", h + 1)
        
        for a in Nodos_industriales:
            net.load.loc[a, "p_mw"] = curva_carga_ind.loc[h, a]
      
        for b in Nodos_residenciales:
            net.load.loc[b, "p_mw"] = curva_carga_res.loc[h, b]
            
        for c in Nodos_comerciales:
            net.load.loc[c, "p_mw"] = curva_carga_com.loc[h, c]
            
        try:
            
            pp.runpp(net)
                
            res_bus = net.res_bus
            res_bus.to_excel(f"Resultado_buses_sin_GD_hora_{h+1}.xlsx", index = True)
            # Extraer la potencia de los otros nodos
            potencia_otros_nodos_sgd = net.res_bus['p_mw'].sum()
            potencia_otros_nodos_sin_gd.append(potencia_otros_nodos_sgd)
            # Extraer la potencia del nodo 30
            potencia_30 = net.res_bus.loc[30, "p_mw"]
            potencia_nodo30_sin_gd.append(potencia_30)
        
        except pp.LoadflowNotConverged:
            print(f"No converge en la hora {h + 1}")
            potencia_nodo30_sin_gd.append(None)
            potencia_otros_nodos_sin_gd.append(None)
                
    # ===============================
    # 🔹 CREACIÓN GD
    # ===============================

    generadores_pv_ind = []
    horas = np.arange(24)
    curvas_pv_ind = {}
    bus_alm_ind = []
    GD_Nodos_ind = 4

    for gen_ind in range (GD_Nodos_ind):
        
        buses_ind = np.random.choice(Nodos_industriales, size = GD_Nodos_ind, replace = False)[0]
        bus_alm_ind.append(buses_ind)
        potencias_pico_MW_ind = np.random.uniform(5.001, 10.0)    
        
        # Crear una curva de generación fotovoltaica para 24 horas (simulación simple)
        
        curva_ind = np.maximum(0, np.sin((horas - 6) * np.pi / 12))  # Modelo básico de generación
        curva_ind = curva_ind / curva_ind.max() * potencias_pico_MW_ind  # Ajustar a la potencia máxima
        curvas_pv_ind[buses_ind] = curva_ind
        curvas_pv_ind = pd.DataFrame(curvas_pv_ind)

    # 📌 Graficar las curvas de generación de todos los generadores
    plt.figure(figsize=(12, 6))

    for gen_i, curva_i in curvas_pv_ind.items():
        plt.plot(horas, curva_i, marker='o', linestyle='-', label=f"Generador industial {gen_i}")

    plt.xlabel("Hora del día")
    plt.ylabel("Generación (MW)")
    plt.title("Curva de Generación Fotovoltaica por Nodo Industrial")
    plt.grid()
    plt.legend()
    # Guardar la gráfica en un archivo
    grafico_archivo = f"Curva de Generación Fotovoltaica por Nodo industrial.png"
    plt.savefig(grafico_archivo)
    plt.close()  # Cerrar la gráfica para que no se muestre
    
    generadores_pv_res = []
    curvas_pv_res = {}
    bus_alm_res = []
    GD_Nodos_res = 4

    for gen_res in range (GD_Nodos_res):
        
        buses_res = np.random.choice(Nodos_residenciales, size = GD_Nodos_res, replace=False)[0]
        bus_alm_res.append(buses_res)
        potencias_pico_MW_res = np.random.uniform(0.001, 0.5)  
        
        # Crear una curva de generación fotovoltaica para 24 horas (simulación simple)
        
        curva_res = np.maximum(0, np.sin((horas - 6) * np.pi / 12))  # Modelo básico de generación
        curva_res = curva_res / curva_res.max() * potencias_pico_MW_res  # Ajustar a la potencia máxima
        curvas_pv_res[buses_res] = curva_res
        curvas_pv_res = pd.DataFrame(curvas_pv_res)
        
    # 📌 Graficar las curvas de generación de todos los generadores
    plt.figure(figsize=(12, 6))

    for gen_r, curva_r in curvas_pv_res.items():
        plt.plot(horas, curva_r, marker='o', linestyle='-', label=f"Generador residencial {gen_r}")

    plt.xlabel("Hora del día")
    plt.ylabel("Generación (MW)")
    plt.title("Curva de Generación Fotovoltaica por Nodo Residencial")
    plt.grid()
    plt.legend()
    # Guardar la gráfica en un archivo
    grafico_archivo = f"Curva de Generación Fotovoltaica por Nodo Residencial.png"
    plt.savefig(grafico_archivo)
    plt.close()  # Cerrar la gráfica para que no se muestre

    generadores_pv_com = []
    curvas_pv_com = {}
    bus_alm_com = []
    GD_Nodos_com = 2

    for gen_com in range (GD_Nodos_com):
        
        buses_com = np.random.choice(Nodos_comerciales, size = GD_Nodos_com, replace=False)[0]
        bus_alm_com.append(buses_com)
        potencias_pico_MW_com = np.random.uniform(0.501, 5)     

        # Crear una curva de generación fotovoltaica para 24 horas (simulación simple)
        
        curva_com = np.maximum(0, np.sin((horas - 6) * np.pi / 12))  # Modelo básico de generación
        curva_com = curva_com / curva_com.max() * potencias_pico_MW_com  # Ajustar a la potencia máxima
        curvas_pv_com[buses_com] = curva_com
        curvas_pv_com = pd.DataFrame(curvas_pv_com)

    # 📌 Graficar las curvas de generación de todos los generadores
    plt.figure(figsize=(12, 6))

    for gen_c, curva_c in curvas_pv_com.items():
        plt.plot(horas, curva_c, marker='o', linestyle='-', label=f"Generador comercial {gen_c}")

    plt.xlabel("Hora del día")
    plt.ylabel("Generación (MW)")
    plt.title("Curva de Generación Fotovoltaica por Nodo Comercial")
    plt.grid()
    plt.legend()
    # Guardar la gráfica en un archivo
    grafico_archivo = f"Curva de Generación Fotovoltaica por Nodo Comercial.png"
    plt.savefig(grafico_archivo)
    plt.close()  # Cerrar la gráfica para que no se muestre

    buses_gd_ind = []
    potencia_gd_ind = []
    buses_gd_res = []
    potencia_gd_res = []
    buses_gd_com = []
    potencia_gd_com = []

    print("--------- Analisis con GD ------------")

    for i in range (24):
        
        print("Hora de analisis: ", i + 1)
        
        for a in Nodos_industriales:
            net.load.loc[a, "p_mw"] = curva_carga_ind.loc[i, a]
      
        for b in Nodos_residenciales:
            net.load.loc[b, "p_mw"] = curva_carga_res.loc[i, b]
            
        for c in Nodos_comerciales:
            net.load.loc[c, "p_mw"] = curva_carga_com.loc[i, c]
        
        net.sgen.drop(net.sgen.index, inplace=True)

        for a in range(curvas_pv_ind.shape[1]):           
                        
            pp.create_sgen(net, bus = bus_alm_ind[a], p_mw = curvas_pv_ind.iloc[i, a], k = 1, type='PV', in_service = True)
            buses_gd_ind.append(bus_alm_ind[a])
            potencia_gd_ind.append(curvas_pv_ind.iloc[i, a])
        
        for b in range(curvas_pv_res.shape[1]):           
                        
            pp.create_sgen(net, bus = bus_alm_res[b], p_mw = curvas_pv_res.iloc[i, b], k = 1, type='PV', in_service = True)
            buses_gd_res.append(bus_alm_res[b])
            potencia_gd_res.append(curvas_pv_res.iloc[i, b])
        
        for c in range(curvas_pv_com.shape[1]):           
                        
            pp.create_sgen(net, bus = bus_alm_com[c], p_mw = curvas_pv_com.iloc[i, c], k = 1, type='PV', in_service = True)
            buses_gd_com.append(bus_alm_com[c])
            potencia_gd_com.append(curvas_pv_com.iloc[i, c])
            
        try:
            
            pp.runpp(net)
            res_bus_GD = net.res_bus
            res_bus_GD.to_excel(f"Resultado_buses_con_GD_hora_{i+1}.xlsx", index = True)
            potencia_30_gd = net.res_bus.loc[30, "p_mw"]
            potencia_nodo30_con_gd.append(potencia_30_gd)
        
        except pp.LoadflowNotConverged:
            print(f"No converge en la hora {i + 1}")
            potencia_nodo30_con_gd.append(None)
            #potencia_otros_nodos_con_gd.append(None)
        

    # CREACIÓN DE DATAFRAMES PARA EXCEL
    df_potencia_nodo30_sin_gd = pd.DataFrame({"Hora": np.arange(1, 25), "Potencia Nodo 30 SGD (MW)": potencia_nodo30_sin_gd})
    df_potencia_nodo30_con_gd = pd.DataFrame({"Hora": np.arange(1, 25), "Potencia Nodo 30 CGD (MW)": potencia_nodo30_con_gd})
    df_potencia_otros_nodos_sin_gd = pd.DataFrame({"Hora": np.arange(1, 25), "Potencia Otros nodos SGD (MW)": potencia_otros_nodos_sin_gd})
    
    # GUARDAR EN UN SOLO ARCHIVO EXCEL CON DOS HOJAS
    with pd.ExcelWriter("Potencia_Nodo30.xlsx") as writer:
        df_potencia_nodo30_sin_gd.to_excel(writer, sheet_name="Sin_GD", index=False)
        df_potencia_nodo30_con_gd.to_excel(writer, sheet_name="Con_GD", index=False)
        df_potencia_otros_nodos_sin_gd.to_excel(writer, sheet_name="ON_SGD", index=False)

    print("Datos guardados en 'Potencia_Nodo30.xlsx' con hojas separadas para cada escenario.")
    
    # Filtrar valores None antes de la suma
    potencia_nodo30_sin_gd_filtrada = [x for x in potencia_nodo30_sin_gd if x is not None]
    potencia_nodo30_con_gd_filtrada = [x for x in potencia_nodo30_con_gd if x is not None]
    potencia_otros_nodos_sin_gd_filtrada = [x for x in potencia_otros_nodos_sin_gd if x is not None]

    # Calcular la suma total de la potencia del nodo 30 en cada escenario
    suma_potencia_sin_gd = sum(potencia_nodo30_sin_gd_filtrada)
    suma_potencia_con_gd = sum(potencia_nodo30_con_gd_filtrada)
    suma_potencia_ON_sin_gd = sum(potencia_otros_nodos_sin_gd_filtrada)

    # Calcular la diferencia absoluta entre ambos escenarios
    diferencia_potencia = abs(suma_potencia_sin_gd - suma_potencia_con_gd)
    Potencia_dia_base = abs(suma_potencia_ON_sin_gd)

    print(f"Suma total sin GD: {suma_potencia_sin_gd:.2f} MWh*día")
    print(f"Suma total con GD: {suma_potencia_con_gd:.2f} MWh*día")
    print(f"Autogeneración FV total: {diferencia_potencia:.2f} MWh*día")
    print(f"Consumo total del sistema: {Potencia_dia_base:.2f} MWh*día")

    # GUARDAR EN DATOS EN LISTAS
    diferencias.append(diferencia_potencia)
    Pbase.append(Potencia_dia_base)

# CALCULAR PROMEDIO DE LAS DIFERENCIAS
promedio_diferencia = np.mean(diferencias)
promedio_Pbase = np.mean(Pbase)

print("\n========= RESULTADOS =========")
print(f"Promedio de Autogeneración FV total: {promedio_diferencia:.2f} MWh*día")
print(f"Promedio de Consumo total del sistema: {promedio_Pbase:.2f} MWh*día")

# CREAR DATAFRAME CON TODAS LAS ITERACIONES
df_diferencias = pd.DataFrame({
    "Iteración": np.arange(1, num_iteraciones + 1),
    "Autogeneración FV total (MWh*día)": diferencias,
    "Consumo total del sistema (MWh*día)": Pbase
})

# AGREGAR EL PROMEDIO AL DATAFRAME
df_promedio_AG = pd.DataFrame({"Promedio Autogeneración díaria (MWh*día)": [promedio_diferencia]})
df_promedio_CT = pd.DataFrame({"Promedio Consumo total del sistema (MWh*día)": [promedio_Pbase]})

# GUARDAR EN ARCHIVO EXCEL
with pd.ExcelWriter("Diferencias_Potencia_Nodo30.xlsx") as writer:
    df_diferencias.to_excel(writer, sheet_name="Iteraciones", index=False)
    df_promedio_AG.to_excel(writer, sheet_name="Promedio", index=False)

# GRAFICAR LOS RESULTADOS: GRÁFICA DE PUNTOS (SCATTER)
plt.figure(figsize=(10, 5))
plt.scatter(df_diferencias["Iteración"], df_diferencias["Autogeneración FV total (MWh*día)"], color="b", label="Autogeneración por Iteración")
plt.xlabel("Iteración")
plt.ylabel("Autogeneración FV total (MWh*día)")
plt.title(f"Diferencia de Potencia en {num_iteraciones} Iteraciones (Puntos)")
plt.legend()
plt.grid(True)
plt.savefig("Diferencia_Potencia_Iteraciones.png")  # GUARDAR IMAGEN
plt.show()

# GRAFICAR HISTOGRAMA DE DIFERENCIAS
plt.figure(figsize=(10, 5))
plt.hist(diferencias, bins=15, color="c", edgecolor="black", alpha=0.7)
plt.xlabel("Autogeneración FV total (MWh*día)")
plt.ylabel("Frecuencia")
plt.title(f"Distribución de la Diferencia de Potencia ({num_iteraciones} Iteraciones)")
plt.grid(True)
plt.savefig("Histograma_Diferencia_Potencia.png")  # GUARDAR IMAGEN
plt.show()

# ========================================================================
#                                  CASOS
# ========================================================================

df_diferencias['Potencia_AG_FV_instalada'] = df_diferencias['Autogeneración FV total (MWh*día)']/4
df_diferencias['h'] = np.random.randint(1, 13, size=len(df_diferencias))
df_diferencias['h_inc'] = np.random.randint(4, 5, size=len(df_diferencias))
df_diferencias['h_r'] = np.random.uniform(1.32, 4, size=len(df_diferencias))
df_diferencias['%_pen'] = np.random.uniform(0.0111, 0.30, size=len(df_diferencias))
df_diferencias['Dt'] = [151.6]* len(df_diferencias)
df_diferencias['PIS'] = abs((df_diferencias['Potencia_AG_FV_instalada']-8935))/8935
df_diferencias['alpha'] = np.random.uniform(0.30, 0.31, size=len(df_diferencias))

# Cálculo de los casos
df_diferencias['Caso Base'] = df_diferencias['Consumo total del sistema (MWh*día)']*df_diferencias['Dt']*1000
df_diferencias['Caso 1 - Sin Cargo de respaldo'] = (df_diferencias['Consumo total del sistema (MWh*día)']-df_diferencias['Autogeneración FV total (MWh*día)'])*df_diferencias['Dt']*1000
df_diferencias['Caso 2 - CRESP'] = ((df_diferencias['Consumo total del sistema (MWh*día)']-df_diferencias['Autogeneración FV total (MWh*día)'])*df_diferencias['Dt']*1000)+(df_diferencias['Potencia_AG_FV_instalada']*df_diferencias['h']*df_diferencias['Dt']*1000)
df_diferencias['Caso 3 - Cargo Respaldo nuevo'] = ((df_diferencias['Consumo total del sistema (MWh*día)']-df_diferencias['Autogeneración FV total (MWh*día)'])*df_diferencias['Dt']*1000)+(df_diferencias['Potencia_AG_FV_instalada']*df_diferencias['h_r']*1*(df_diferencias['alpha']*df_diferencias['Dt']*1000))+(df_diferencias['%_pen']*(df_diferencias['Potencia_AG_FV_instalada']*df_diferencias['h_inc']*(df_diferencias['alpha']*df_diferencias['Dt']*1000)))

df_diferencias.to_excel("Analisis de casos.xlsx", sheet_name="Casos", index=False)

# Calcular los promedios de cada caso
promedio_base = df_diferencias["Caso Base"].mean()
promedio_caso1 = df_diferencias["Caso 1 - Sin Cargo de respaldo"].mean()
promedio_caso2 = df_diferencias["Caso 2 - CRESP"].mean()
promedio_caso3 = df_diferencias["Caso 3 - Cargo Respaldo nuevo"].mean()

# Graficar los casos vs iteraciones
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df_diferencias["Iteración"], df_diferencias["Caso Base"], marker='o', linestyle='-', label="Caso Base")
ax.plot(df_diferencias["Iteración"], df_diferencias["Caso 1 - Sin Cargo de respaldo"], marker='s', linestyle='-', label="Caso 1 - Sin Cargo de respaldo")
ax.plot(df_diferencias["Iteración"], df_diferencias["Caso 2 - CRESP"], marker='^', linestyle='-', label="Caso 2 - CRESP")
ax.plot(df_diferencias["Iteración"], df_diferencias["Caso 3 - Cargo Respaldo nuevo"], marker='x', linestyle='-', label="Caso 3 - Cargo Respaldo nuevo")

# Agregar líneas punteadas del promedio
ax.axhline(promedio_base, color='blue', linestyle='dashed', alpha=0.7, label="Promedio Caso Base")
ax.axhline(promedio_caso1, color='orange', linestyle='dashed', alpha=0.7, label="Promedio Caso 1")
ax.axhline(promedio_caso2, color='green', linestyle='dashed', alpha=0.7, label="Promedio Caso 2")
ax.axhline(promedio_caso3, color='purple', linestyle='dashed', alpha=0.7, label="Promedio Caso 3")

# Agregar etiquetas de los valores promedio en la gráfica
ax.text(num_iteraciones + 0.15, promedio_base, f'{promedio_base/1e6:.2f} M COP', color='blue', verticalalignment='center')
ax.text(num_iteraciones + 0.15, promedio_caso1, f'{promedio_caso1/1e6:.2f} M COP', color='orange', verticalalignment='center')
ax.text(num_iteraciones + 0.15, promedio_caso2, f'{promedio_caso2/1e6:.2f} M COP', color='green', verticalalignment='center')
ax.text(num_iteraciones + 0.15, promedio_caso3, f'{promedio_caso3/1e6:.2f} M COP', color='red', verticalalignment='center')

# Configurar la gráfica
plt.xlabel("Número de iteraciones (día)")
plt.ylabel("Recaudo por distribución (100M COP)")
plt.title("Comparación de Casos de estudio en un año natural")
plt.legend()
plt.grid(True)
plt.savefig("Comparación_de_casos_de_estudio.png")  # GUARDAR IMAGEN
plt.show()
