<template>
  <div class="image-search-result">
    <div class="isr-header">
      <h4><PictureOutlined /> 图片搜索结果</h4>
      <div class="result-summary">找到 {{ items.length }} 件相似文物</div>
    </div>
    <div class="isr-grid">
      <div v-for="(item, index) in items" :key="index" class="isr-card">
        <div class="isr-image-wrapper" v-if="item.imageUrl">
          <img
            :src="getFullUrl(item.imageUrl)"
            :alt="item.name"
            class="isr-image"
            @error="onImgError($event)"
          />
        </div>
        <div class="isr-info">
          <div class="isr-name">{{ item.name }}</div>
          <div class="isr-meta" v-if="item.museum">{{ item.museum }}</div>
          <div class="isr-score" v-if="item.score">相似度: {{ item.score }}</div>
          <div class="isr-desc" v-if="item.desc">{{ item.desc }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { PictureOutlined } from '@ant-design/icons-vue'

const props = defineProps({
  data: {
    type: String,
    required: true
  }
})

const getFullUrl = (url) => {
  if (url.startsWith('http')) return url
  return `${import.meta.env.VITE_API_BASE_URL || ''}${url}`
}

const onImgError = (e) => {
  e.target.style.display = 'none'
}

// 解析工具返回的纯文本为结构化数据
const items = computed(() => {
  const text = typeof props.data === 'string' ? props.data : String(props.data)
  const results = []

  // 按【N】分割条目
  const entryPattern = /【(\d+)】([^【]*)/g
  let match
  while ((match = entryPattern.exec(text)) !== null) {
    const block = match[2].trim()
    const item = { name: '', museum: '', score: '', imageUrl: '', desc: '' }

    // 提取名称和元信息：名称（博物馆，相似度: 0.xxxx）
    const headerMatch = block.match(/^(.+?)（(.+?)，相似度:\s*([\d.]+)）/)
    if (headerMatch) {
      item.name = headerMatch[1].trim()
      item.museum = headerMatch[2].trim()
      item.score = headerMatch[3].trim()
    }

    // 提取描述
    const descMatch = block.match(/描述:\s*(.+?)(?=\n|$)/)
    if (descMatch) {
      item.desc = descMatch[1].trim()
    }

    // 提取图片 URL：![name](url) 格式
    const imgMatch = block.match(/!\[.*?\]\((.+?)\)/)
    if (imgMatch) {
      item.imageUrl = imgMatch[1].trim()
    }

    if (item.name) {
      results.push(item)
    }
  }

  return results
})
</script>

<style lang="less" scoped>
.image-search-result {
  background: var(--gray-0);
  border-radius: 8px;

  .isr-header {
    padding: 12px 16px;
    background: var(--gray-25);

    h4 {
      margin: 0 0 4px 0;
      color: var(--gray-800);
      font-size: 14px;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 6px;

      .anticon {
        color: var(--main-color);
        font-size: 13px;
      }
    }

    .result-summary {
      font-size: 12px;
      color: var(--gray-500);
    }
  }

  .isr-grid {
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .isr-card {
    display: flex;
    gap: 12px;
    padding: 10px;
    border: 1px solid var(--gray-150);
    border-radius: 8px;
    background: var(--gray-0);
    transition: all 0.2s ease;

    &:hover {
      background: var(--gray-25);
    }
  }

  .isr-image-wrapper {
    flex-shrink: 0;
    width: 100px;
    height: 100px;
    border-radius: 6px;
    overflow: hidden;
    background: var(--gray-50);

    .isr-image {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
  }

  .isr-info {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;

    .isr-name {
      font-size: 14px;
      font-weight: 600;
      color: var(--gray-900);
    }

    .isr-meta {
      font-size: 12px;
      color: var(--gray-500);
    }

    .isr-score {
      font-size: 11px;
      color: var(--main-700);
      background: var(--main-5, rgba(74, 144, 164, 0.05));
      padding: 1px 6px;
      border-radius: 4px;
      width: fit-content;
    }

    .isr-desc {
      font-size: 12px;
      color: var(--gray-600);
      line-height: 1.5;
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
    }
  }
}
</style>
