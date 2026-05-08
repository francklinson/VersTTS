#!/bin/bash
# 生成自签名 SSL 证书
# 用法: ./scripts/generate_ssl.sh [输出目录]

OUTPUT_DIR="${1:-ssl}"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "      生成自签名 SSL 证书"
echo "========================================"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo ""

openssl req -x509 -newkey rsa:2048 \
    -keyout "$OUTPUT_DIR/key.pem" \
    -out "$OUTPUT_DIR/cert.pem" \
    -days 365 \
    -nodes \
    -subj "/C=CN/ST=State/L=City/O=VersTTS/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "      SSL 证书生成成功"
    echo "========================================"
    echo "证书文件: $OUTPUT_DIR/cert.pem"
    echo "私钥文件: $OUTPUT_DIR/key.pem"
    echo ""
    echo "启用 HTTPS:"
    echo "  编辑 start_server.sh，设置:"
    echo "    SSL_CERT=\"$OUTPUT_DIR/cert.pem\""
    echo "    SSL_KEY=\"$OUTPUT_DIR/key.pem\""
    echo ""
    echo "然后启动服务:"
    echo "  bash start_server.sh start"
    echo ""
    echo "注意: 浏览器会提示证书不受信任，点击「继续访问」即可。"
else
    echo "错误: 证书生成失败"
    exit 1
fi
