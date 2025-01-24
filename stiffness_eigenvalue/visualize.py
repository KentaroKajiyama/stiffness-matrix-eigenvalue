import matplotlib.pyplot as plt
import numpy as np
import time
##################################################
SWITCH = 250
###################################################
# フレームワークの可視化用
def custom_visualize(framework,label, limit=False):
  fig, ax = plt.subplots()
  ax.scatter(framework.coordinates[:,0], framework.coordinates[:,1], c='blue')
  # プロットの設定
  ax.axhline(0, color='grey', linewidth=0.5)
  ax.axvline(0, color='grey', linewidth=0.5)
  ax.grid(True, linestyle='--', alpha=0.7)
  # 角度データを生成
  theta = np.linspace(0, 2 * np.pi, 100)
  # 半径0.33と0.66の円のデータを生成
  radii = [0.31622, 0.63244]
  circles = [(r * np.cos(theta), r * np.sin(theta)) for r in radii] 
  # 円をプロット
  for i, (x, y) in enumerate(circles):
    ax.plot(x, y, label=f'Circle (r={radii[i]})')
  # フレームワークの各点のプロット
  for bond in framework.bonds:
    start, end = framework.coordinates[bond]
    ax.plot([start[0], end[0]], [start[1], end[1]], 'k-') 
  # 固定された節点を赤色で表示
  for pin in framework.pins:
    ax.scatter(framework.coordinates[pin,0], framework.coordinates[pin,1], c='red', marker='D')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title("Custom Visualization of the Framework, {}".format(label))
  plt.legend(loc="best")
  fig.canvas.mpl_connect("key_press_event", on_key)
  plt.show(block=False)
  if limit:
    plt.show(block=False)
    time.sleep(10)
    plt.close(fig)
  else:
    plt.show()
# 固有値のプロット 横軸iteration回数、縦軸固有値
def plot_eigen_vals(eigen_vals, limit=False):
  fig = plt.figure(figsize=(6,4))
  ax1 = fig.add_subplot(1,1,1)
  ax1.plot(eigen_vals, label="eigenvalues")
  ax1.set_xlabel("index")
  ax1.set_ylabel("eigenvalue")
  ax1.legend(loc="best")
  fig.canvas.mpl_connect("key_press_event", on_key)
  if limit:
    plt.show(block=False)
    time.sleep(10)
    plt.close(fig)
  else:
    plt.show()
# Armijoの条件のalphaのプロット、横軸試行回数、縦軸alpha値
def plot_alpha(alpha_box, limit=False):
  fig = plt.figure(figsize=(6,4))
  ax1 = fig.add_subplot(1,1,1)
  ax1.plot(alpha_box, label="alpha")
  ax1.set_xlabel("index")
  ax1.set_ylabel("alpha")
  ax1.legend(loc="best")
  fig.canvas.mpl_connect("key_press_event", on_key)
  if limit:
    plt.show(block=False)
    time.sleep(10)
    plt.close(fig)
  else:
    plt.show()

import matplotlib.pyplot as plt
import time

def plot_eigen_vals_and_alpha(eigen_vals, alpha_box, mulitiplicity_vec_box, limit=False, is_save=False, save_prefix="plot"):
  fig = plt.figure(figsize=(10,6))
  fig2 = plt.figure(figsize=(6,4))

  ax1 = fig.add_subplot(1,2,1)
  ax1.plot(eigen_vals, label="eigenvalues")
  ax1.set_xlabel("index")
  ax1.set_ylabel("eigenvalue")
  ax1.legend(loc="best")

  ax2 = fig.add_subplot(1,2,2)
  ax2.plot(alpha_box, label="alpha")
  ax2.set_xlabel("index")
  ax2.set_ylabel("alpha")
  ax2.legend(loc="best")

  ax3 = fig2.add_subplot(1,1,1)
  ax3.plot(eigen_vals, label="eigenvalues")
  ax3.plot(alpha_box, label="alpha")
  ax3.plot(mulitiplicity_vec_box, label="multiplicity")

  ax3.set_xlabel("index")
  ax3.set_ylabel("eigenvalue and alpha")
  ax3.legend(loc="best")

  if is_save:
    fig.savefig(f"{save_prefix}_eigen_alpha.png", dpi=300)
    fig2.savefig(f"{save_prefix}_combined.png", dpi=300)
    print(f"Figures saved as {save_prefix}_eigen_alpha.png and {save_prefix}_combined.png")
    plt.close(fig)
    plt.close(fig2)
  else:
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig2.canvas.mpl_connect("key_press_event", on_key)

    if limit:
      plt.show(block=False)
      time.sleep(10)
      plt.close(fig)
      plt.close(fig2)
    else:
      plt.show()

# グラフを閉じる用
def on_key(event):
  if event.key == 'enter':
    plt.close(event.canvas.figure)