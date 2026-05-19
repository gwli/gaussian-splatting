# WebXR Viewer for PICO 4 Ultra

在 VR 头显里浏览训练好的 3D Gaussian Splatting 场景。专门为 **PICO 4 Ultra** 调优，但任何支持 WebXR 的浏览器/头显都能用。

底层使用 [mkkellogg/GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)（Three.js + WebXR，MIT 协议）。

---

## 一、当前部署状态（本服务器已就绪）

服务正在 `webxr-server` 容器里运行：

| 项 | 值 |
|---|---|
| 服务器 IP | `10.176.199.45` |
| 端口 | `8443` (HTTPS，WebXR 必须 HTTPS) |
| 证书 | 自签名，10 年有效期 |

### 在 PICO 4 Ultra 上打开（用户操作）

1. 戴上头显，连接到 **与服务器同一个 WiFi**
2. 启动 **PICO 浏览器**（PICO Browser，不是其他应用内浏览器）
3. 地址栏输入：
   ```
   https://10.176.199.45:8443/viewer.html?scene=scene_021
   ```
4. 弹出"证书不受信任"警告 → 点 **"高级" → "继续前往"**（自签名证书的正常提示）
5. 等待 PLY 下载（50–80 MB，几秒钟）
6. 场景加载完成后，**屏幕右下角会出现 `Enter VR` 按钮**，点击它
7. 完成！可以在场景中 6DoF 自由漫游

### 可切换的场景

| 链接 | 场景 |
|---|---|
| `https://10.176.199.45:8443/viewer.html?scene=scene_021` | 8K 全景，5 分钟航拍 (49M) |
| `https://10.176.199.45:8443/viewer.html?scene=scene_022` | 5:38 航拍 (64M) |
| `https://10.176.199.45:8443/viewer.html?scene=scene_023` | 6:20 航拍 (17M，加载最快) |
| `https://10.176.199.45:8443/viewer.html?scene=scene_025` | 9:21 航拍 (74M) |

场景内底部会有切换条，可以直接点击切换。也支持自定义 URL：

```
https://10.176.199.45:8443/viewer.html?ply=https://yourserver/your.ply
```

### VR 模式控制

进入 VR 后用 PICO 4 Ultra 手柄：
- **扳机键 (Trigger)**：选择/交互
- **摇杆 (Thumbstick)**：平滑移动（前/后/左右走）
- **侧握键 (Grip)**：可选，抓握后平移整个场景

也可以摘下头显用电脑/手机访问，**鼠标拖拽 + 滚轮**控制视角，不进入 VR 也能预览。

---

## 二、从零部署（在另一台服务器上）

### 前提
- Docker + nvidia-container-toolkit（GPU 用于训练，本 viewer 只用 CPU 渲染）
- Linux 主机，可访问 GitHub
- 已有训练好的 `.ply` 点云

### 步骤

```bash
# Step 1: 克隆 viewer 源码
cd /path/to/gaussian-splatting/webxr_viewer
git clone https://github.com/mkkellogg/GaussianSplats3D.git

# Step 2: 构建 viewer (用 node:20 容器，约 2 分钟)
docker run --rm --user 0:0 \
    -v $(pwd)/GaussianSplats3D:/app -w /app node:20 \
    bash -c "npm install --silent && npm run build"

# Step 3: 拷贝自定义 viewer.html 到构建目录
cp viewer.html GaussianSplats3D/build/demo/viewer.html

# Step 4: 生成自签名证书 (10 年期，绑定本机 IP)
mkdir -p certs && cd certs
HOST_IP=$(hostname -I | awk '{print $1}')
openssl req -x509 -nodes -days 3650 \
    -newkey rsa:2048 \
    -keyout server.key -out server.crt \
    -subj "/CN=gaussian-splat-webxr" \
    -addext "subjectAltName=IP:$HOST_IP,IP:127.0.0.1,DNS:localhost"
cd ..

# Step 5: 启动 HTTPS 容器
docker run -d --name webxr-server -p 8443:8443 \
    -v $(pwd)/GaussianSplats3D/build/demo:/viewer:ro \
    -v /path/to/your/data:/scenes:ro \
    -v $(pwd)/certs:/certs:ro \
    -v $(pwd)/nginx-webxr.conf:/etc/nginx/conf.d/default.conf:ro \
    nginx:alpine

# Step 6: 验证
curl -sk https://localhost:8443/viewer.html | head -3
```

### 调整 viewer 默认路径

编辑 `viewer.html` 中的这一行来匹配你的目录结构：

```javascript
const sceneUrl = customUrl ||
  `/data/8kpano/scenes/${scene}/output/point_cloud/iteration_30000/point_cloud.ply`;
```

`/data/` 对应 nginx 容器里的 `/scenes/`（即上面 `-v /path/to/your/data:/scenes:ro` 挂载的目录）。

---

## 三、服务管理

### 查看状态
```bash
docker ps --filter "name=webxr-server"
docker logs webxr-server --tail 20
```

### 重启
```bash
docker restart webxr-server
```

### 停止 / 删除
```bash
docker stop webxr-server && docker rm webxr-server
```

### 更新 viewer.html
修改 `viewer.html` 后：
```bash
cp viewer.html GaussianSplats3D/build/demo/viewer.html
# 不需要重启容器，nginx 直接读挂载的文件
```

### 添加新场景
把训练好的 `.ply` 放到 `/raid/git/gaussian-splatting/data/...` 下，然后访问：
```
https://10.176.199.45:8443/viewer.html?ply=/data/path/to/file.ply
```
不用重启服务。

---

## 四、架构

```
┌──────────────────────────────┐
│ PICO 4 Ultra                 │
│ ├─ PICO Browser              │
│ └─ WebXR API (immersive-vr)  │
└────────────┬─────────────────┘
             │ HTTPS :8443
             ▼
┌──────────────────────────────────────┐
│ webxr-server (nginx:alpine)          │
│                                       │
│  /viewer.html  → 自定义 viewer        │
│  /lib/         → Three.js + GS lib    │
│  /data/        → PLY 点云 (静态文件)  │
│                                       │
│ CORS + COOP/COEP/CORP 头已配置好     │
└──────────────────────────────────────┘
             │
             └─ Mount: /raid/git/gaussian-splatting/data
```

---

## 五、Troubleshooting

| 现象 | 原因 / 解决 |
|---|---|
| 浏览器打不开 (Site can't be reached) | 头显和服务器不在同一 WiFi |
| "您的连接不是私密连接" 警告 | 自签名证书的正常提示，点"高级 → 继续前往" |
| 加载完成但没有 "Enter VR" 按钮 | 必须用 `https://` 不能用 `http://` |
| "WebXR not supported" | 请用 PICO 自带浏览器（不是第三方浏览器） |
| Enter VR 后画面黑屏 | 先在普通模式下拖动鼠标确认场景已加载 |
| 加载非常慢 | PLY 文件较大，第一次访问需缓存到本地 |
| 想用真证书替代自签名 | 用 Let's Encrypt + DNS 域名，把 `certs/server.crt` 和 `certs/server.key` 替换即可 |
| 端口 8443 被占用 | 改 `nginx-webxr.conf` 里的 `listen 8443`，并改 docker `-p` 端口映射 |

### WebXR 兼容性自检
在 PICO 浏览器里访问 https://immersiveweb.dev/ → 点 "Enter VR" 测试。能成功就说明头显配置 OK，问题在我们的服务端。

---

## 六、安全提示

- **自签名证书只适合内网/开发**，不要暴露到公网
- 当前配置 CORS 是 `*`（任意来源），生产环境改为白名单
- 私钥 `certs/server.key` 不要提交到 git（已加入 `.gitignore`）
- 如果场景内容敏感，给 nginx 加 basic auth 或反向代理鉴权
