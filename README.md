# superpicky-coreml

把 [SuperPicky](https://github.com/jamesphotography/SuperPicky) 所用的模型转换成 Core ML（fp16）并打包发布。非 SuperPicky 官方项目。

| 模型 | 用途 | 许可证 |
|---|---|---|
| `yolo11l-seg` | 鸟类检测与分割 | AGPL-3.0 |
| `birdtrace` | 鸟种识别 | CC BY-NC 4.0 |
| `topiq` | 美学评分 | CC BY-NC-SA 4.0 |
| `keypoint` | 鸟眼、鸟喙关键点（SuperEyes） | GPL-3.0 |
| `flight` | 飞版判断（SuperFlier） | GPL-3.0 |

各模型的许可证和署名见 `models/<模型名>/`，转换代码为 AGPL-3.0（`archs.py` 中的 TOPIQ 部分除外）。

## 使用

需要 macOS 和 [uv](https://docs.astral.sh/uv/)，在仓库根目录执行：

```sh
(cd tools/modelconv && uv sync)
uv run --project tools/modelconv python -m modelconv.fetch_models   # 下载权重到 models/
uv run --project tools/modelconv python -m modelconv.convert_all    # 转换到 models/coreml/
python3 tools/package_models.py                                      # 打包到 dist/（zip 和 manifest.json）
```

更新鸟名表（需要 SuperPicky 仓库）：

```sh
python3 tools/export_species_names.py --db ../SuperPicky/birdid/data/bird_reference.sqlite \
    --out models/birdtrace/species_names.json
```

## 发布

推送 `v*` tag，GitHub Actions 会自动转换、打包并发布到同名 Release：

```sh
git tag v1.0.0 && git push origin v1.0.0
```

同一个提交重复构建，产物逐字节相同。

## 测试

```sh
cd tools/modelconv
uv run ruff check . && uv run pytest
```
