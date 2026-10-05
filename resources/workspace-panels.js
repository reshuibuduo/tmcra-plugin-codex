'use strict';
(() => {
  const pages = new Map();
  const labels = { confirmed:'已确认',provisional:'待验证',superseded:'已更新',open:'待解决',learned:'学习知识',project:'项目知识',personal:'个人信息',user:'用户',assistant:'助手',goal:'目标',requirement:'需求',decision:'决定',action:'行动',result:'结果',problem:'问题',solution:'方案',lesson:'经验',preference:'偏好',fact:'事实',open_question:'待解决',depends_on:'依赖',resolves:'解决',updates:'更新',supersedes:'替代',contradicts:'存在矛盾',reinforces:'相互支持',derived_from:'来源于',applies_to:'应用于',related:'相关',leads_to:'推动',continues:'延续',branches:'分支',converges:'汇合',contains:'包含',parent:'上级',supports:'支持'};
  const word = (row, key) => String(row?.display?.zh?.[key] || row?.[key] || '');
  const heading = (root, eyebrow, title, description) => { const h=el('div',undefined,root,'page-head');const box=el('div',undefined,h);el('div',eyebrow,box,'eyebrow');el('h1',title,box);el('p',description,box);return h; };
  function fail(root,error,retry){root.replaceChildren();const box=el('div',undefined,root,'workspace-error');box.setAttribute('role','alert');el('p',errorText(error),box);button('重试',box,retry,{style:'btn small'});}
  function field(parent,label,id,type='text'){const box=el('label',label,parent,'field');const input=el('input',undefined,box);input.id=id;input.type=type;input.autocomplete='off';input.required=type!=='password';input.setAttribute('aria-label',label);return input;}
  async function localModels(root){
    const value=await api('local_deployment_status');if(!root.isConnected)return;root.replaceChildren();
    const head=el('header',undefined,root);el('h2','把记忆系统放在这台电脑',head);pill(value.ready?'本机服务已就绪':'本地部署 · 预览版',head);
    el('p','向量检索、重排、记忆写入与后台整理均在本机运行。使用独立身份和数据目录，现有云端记忆保留。',root);
    el('small',`${value.requirement}。检测到 ${value.ramGiB}GB 内存；推荐基于容量估算。`,root);
    if(value.missing)el('p',value.missing,root,'workspace-status');
    const grid=el('div',undefined,root,'local-model-grid');const busy=['installing','starting'].includes(value.operation.state);
    for(const profile of value.profiles){
      const card=el('article',undefined,grid,'local-model-card');const recommended=profile.id===value.recommendedProfile;
      if(recommended)pill('本机推荐',card);el('h3',profile.name_zh,card);el('p',profile.recommendation_zh,card);
      el('strong',profile.embedding.repo_id.split('/').pop(),card);el('small',`Embedding · ${profile.embedding.dimensions} 维`,card);
      el('strong',profile.reranker.repo_id.split('/').pop(),card);el('small','Reranker · 检索结果重排',card);
      el('p',`检索权重约 ${(profile.weights_bytes/1e9).toFixed(2)}GB · 完整系统建议 ${profile.system_ram_gib_recommended_for_full_memory}GB 内存`,card);
      el('small',`${profile.embedding.license} / ${profile.reranker.license}；生成模型另需约 2.50GB。`,card);
      if(profile.id!=='lite-cpu')el('small','此档位尚待对应硬件的完整实测。',card);
      else el('small','CPU 写入和原文召回已测；复杂编译与后台整理尚待验收。',card);
      button(value.installedProfile===profile.id?'重新准备此档位':'一键安装',card,()=>openEditor({kind:'confirm',title:`安装${profile.name_zh}？`,
        note:`首次安装会下载检索权重、2.50GB 生成模型和 Python 运行环境，保存在 ${value.dataRoot}。将自动切换到本机记忆，无需 TMCRA 账号和服务器。原云端配置与记忆保留；安装失败时保持本地模式。完成后重启宿主即可自动接入。此版本支持 Windows x64。`,
        confirm:'确认下载并安装',save:async()=>{await api('local_deployment_install',{profile:profile.id});await localModels(root);}}),
        {style:recommended?'btn primary':'btn',disabled:!value.available||busy||value.running||value.ramGiB<profile.system_ram_gib_min});
    }
    const status=el('p',value.operation.error||(value.ready?'本地服务已通过启动检查。':'状态：'+({idle:'尚未启动',installing:'正在安装',starting:'正在启动与检查',failed:'安装失败',ready:'已就绪'}[value.operation.state]||value.operation.state)),root,'workspace-status');status.setAttribute('role','status');
    const actions=el('div',undefined,root,'provider-actions');button('刷新本地状态',actions,()=>localModels(root),{style:'btn small',symbol:'refresh'});
    if(value.installedProfile&&!value.running)button('启动本地服务',actions,async()=>{await api('local_deployment_start');await localModels(root);},{style:'btn small',disabled:busy||!value.available});
    if(value.running)button('停止本地服务',actions,()=>openEditor({kind:'confirm',title:'停止本地记忆服务？',note:'本机写入与召回将暂停，模型和已保存的记忆保留。',confirm:'停止服务',save:async()=>{await api('local_deployment_stop');await localModels(root);}}),{style:'btn small'});
    if(value.connectionConfig){el('p','独立本地身份已生成并自动选用。重启宿主后即可接入，无需填写账号或 API Key。',root);const path=el('code',value.connectionConfig,root,'local-config-path');}
    if(busy&&!value.ready)setTimeout(()=>{if(root.isConnected&&!$('page-providers').hidden)void localModels(root).catch(error=>fail(root,error,()=>localModels(root)));},4000);
  }
  async function providers(root){
    const current=await api('providers_read');root.replaceChildren();
    heading(root,'MODELS, ON YOUR TERMS','模型配置','为记忆写入与后台整理分别选择 API，密钥保存在本机。');
    const local=el('section',undefined,root,'panel local-deployment');
    void localModels(local).catch(error=>fail(local,error,()=>localModels(local)));
    const form=el('form',undefined,root,'provider-form');form.autocomplete='off';
    const grid=el('div',undefined,form,'provider-grid');const stages={};let dirty=false;
    for(const stage of ['writer','organizer']){
      const card=el('section',undefined,grid,'panel provider-stage');const header=el('header',undefined,card);
      el('h2',stage==='writer'?'记忆写入':'后台整理',header);pill(stage==='writer'?'WRITER':'ORGANIZER',header);
      el('p',stage==='writer'?'从对话中提取记忆，保存事实与来源。':'整理已有记忆、更新长期关系，支持独立配置模型。',card);
      let inherit;
      if(stage==='organizer'){const label=el('label',undefined,card,'inherit-label');inherit=el('input',undefined,label);inherit.type='checkbox';inherit.id='provider-inherit';inherit.checked=current.organizer?.inheritWriter!==false;el('span','沿用记忆写入的 API 和模型',label);}
      const fields=el('div',undefined,card);const providerLabel=el('label','接口类型',fields,'field');const provider=el('select',undefined,providerLabel);provider.id=stage+'-provider';provider.setAttribute('aria-label',stage+' 接口类型');
      for(const [value,name] of [['openai-compatible','OpenAI 兼容接口'],['deepseek','DeepSeek'],['local-openai-compatible','本机兼容接口']]){const o=el('option',name,provider);o.value=value;}
      const base=field(fields,'API 地址',stage+'-base');base.placeholder='https://provider.example/v1';
      const model=field(fields,'模型名称',stage+'-model');model.placeholder='填写服务商提供的模型 ID';
      const key=field(fields,'API Key',stage+'-key','password');key.placeholder=current[stage]?.credentialPresent?'密钥已保存；留空保留':'仅保存在当前电脑';key.autocomplete='new-password';
      el('small','切换 API 地址后，需要重新填写该接口的密钥。',fields);
      const old=current[stage]||{};provider.value=old.provider||'openai-compatible';base.value=old.baseUrl||'';model.value=old.model||'';
      const actions=el('div',undefined,card,'provider-actions');const status=el('p',undefined,card,'provider-test');status.id=stage+'-test-result';status.setAttribute('role','status');
      button('测试推理',actions,async()=>{if(!form.reportValidity())return;status.textContent='正在使用虚构样本测试，不会发送真实记忆…';try{const result=await api('providers_test',{stage,config:payload()});status.textContent=`测试通过 · ${result.latencyMs} ms\n实际响应模型：${result.servedModel}\nJSON 输出校验通过；尚未执行正式记忆作业。`;}catch(error){status.textContent='测试失败：'+errorText(error);throw error;}},{symbol:'activity'});
      const clear=button('移除密钥',actions,()=>openEditor({kind:'confirm',title:'移除本机模型密钥？',note:stage==='writer'?'沿用此配置的后台整理也会失去访问凭据。':'仅移除后台整理的本机密钥。',confirm:'确认移除',save:async()=>{await api('providers_clear',{stage});dirty=false;await providers(root);}}),{style:'btn text',disabled:!old.credentialPresent||Boolean(inherit?.checked)});
      stages[stage]={provider,base,model,key,inherit,fields,clear};
      if(inherit){const update=()=>{fields.hidden=inherit.checked;fields.querySelectorAll('input,select').forEach(n=>n.disabled=inherit.checked);clear.disabled=inherit.checked||!old.credentialPresent;};inherit.onchange=update;update();}
    }
    const savebar=el('div',undefined,form,'provider-savebar');const saveNote=el('p','密钥只在本机输入和保存。测试会向所填服务商发送少量虚构内容。',savebar);
    const save=el('button','保存配置',savebar,'btn primary');save.type='submit';save.id='saveProviders';
    function payload(){const get=s=>({provider:s.provider.value,baseUrl:s.base.value.trim(),model:s.model.value.trim(),...(s.key.value.trim()?{apiKey:s.key.value.trim()}:{})});return {writer:get(stages.writer),organizer:stages.organizer.inherit.checked?{inheritWriter:true}:{inheritWriter:false,...get(stages.organizer)}};}
    form.oninput=()=>{dirty=true;saveNote.textContent='有尚未保存的配置。测试使用当前填写内容。';};
    form.onsubmit=event=>{event.preventDefault();if(!form.reportValidity())return;run(async()=>{const value=await api('providers_save',{config:payload()});for(const stage of Object.values(stages))stage.key.value='';dirty=false;await providers(root);note(value.configured?'模型配置已保存到本机':'配置已更新');},save);};
    const oldHandler=pages.get('providerUnload');if(oldHandler)window.removeEventListener('beforeunload',oldHandler);
    const before=event=>{if(dirty){event.preventDefault();event.returnValue='';}};pages.set('providerUnload',before);window.addEventListener('beforeunload',before);
  }
  function scopePicker(root,onChange){const toolbar=el('div',undefined,root,'workspace-toolbar');const label=el('label','记忆范围',toolbar);const select=el('select',undefined,label);select.setAttribute('aria-label','记忆范围');for(const item of data.availableScopes||[{scope:data.scope,label:'当前项目'}]){const o=el('option',item.label,select);o.value=item.scope;}select.value=(data.availableScopes||[]).find(x=>x.label==='个人全局')?.scope||data.scope;select.onchange=()=>onChange(select.value);return {toolbar,select};}
  function projectionStatus(root,value){el('p',`${value.projection_state==='ready'?'已整理视图':'基础视图 · 等待后台整理'}${value.stale?' · 内容有更新，当前投影待刷新':''}。内容与关系来自服务端，原始记录保留用于核对。`,root,'workspace-status');}
  async function evidence(container,node,scope){
    container.replaceChildren();el('h3','原始证据',container);
    const ids=[...new Set(node.source_record_ids?.length?node.source_record_ids:[node.source_record_id||node.memory_id].filter(Boolean))];
    if(!ids.length){el('p','此条目尚未提供可查询的来源标识。',container);return;}
    for(const id of ids){const block=el('div',undefined,container);const load=async cursor=>{
      const result=await api('evidence',{scope,memory_id:id,...(cursor?{cursor}:{})});
      for(const item of result.items||[]){const article=el('article',undefined,block,'knowledge-claim');pill(labels[item.actor_role]||item.role||'来源',article);el('p',item.text,article);el('div',item.source_record_id,article,'source-id mono');
        if(item.source_record_id&&data.policy.write){const actions=el('div',undefined,article,'knowledge-evidence');button('纠正此来源',actions,()=>sourceFeedback('correct',{memory_id:item.source_record_id,content:item.text},{scope},{}),{style:'btn small'});}
      }
      if(result.page?.next_cursor){const next=button('加载更多来源',block,async()=>{await load(result.page.next_cursor);next.remove();},{style:'btn small'});}
    };await load();}
  }
  async function knowledge(root){
    root.replaceChildren();heading(root,'KNOWLEDGE THAT STAYS WITH YOU','知识库','把学过的知识、项目决定与个人偏好，整理成可追溯的条目。');
    let loaded,selected=null,offset=0;const scope=scopePicker(root,()=>load());const input=el('input',undefined,scope.toolbar);input.type='search';input.placeholder='搜索知识标题、事实与正文';input.setAttribute('aria-label','搜索知识库');
    const collection=el('select',undefined,scope.toolbar);collection.setAttribute('aria-label','知识分类');for(const [v,t] of [['','全部分类'],['learned','学习知识'],['project','项目知识'],['personal','个人信息']]){const o=el('option',t,collection);o.value=v;}
    button('刷新知识库',scope.toolbar,()=>load(),{symbol:'refresh'});const content=el('div',undefined,root);let generation=0;
    async function load(){const seq=++generation;content.replaceChildren();el('p','正在读取知识库…',content,'workspace-status');try{const value=await api('knowledge',{scope:scope.select.value});if(seq!==generation)return;if(!Array.isArray(value.pages))throw Error('服务尚未返回知识库页面，请检查服务版本与整理状态。');loaded=value;selected=null;offset=0;render();}catch(error){if(seq===generation)fail(content,error,load);}}
    function render(){content.replaceChildren();projectionStatus(content,loaded);const q=input.value.trim().toLowerCase();const filtered=loaded.pages.filter(p=>(!collection.value||p.collection===collection.value)&&(!q||JSON.stringify(p).toLowerCase().includes(q)));
      if(!filtered.length){empty(content,'暂时没有匹配的知识',loaded.pages.length?'调整搜索词或分类，查看其他条目。':'记忆进入后台整理后，知识条目会出现在这里。');return;}
      const list=filtered.slice(offset,offset+40);if(!list.some(p=>p.page_id===selected?.page_id))selected=list[0];const layout=el('div',undefined,content,'knowledge-layout');const index=el('div',undefined,layout,'knowledge-index');index.setAttribute('aria-label','知识条目');const article=el('article',undefined,layout,'panel knowledge-article');
      for(const p of list){const b=button('',index,()=>{selected=p;render();},{style:'knowledge-entry'});b.setAttribute('aria-current',String(p.page_id===selected.page_id));b.replaceChildren();el('strong',word(p,'title'),b);el('p',word(p,'abstract').slice(0,95),b);el('small',labels[p.collection]||p.collection||'知识条目',b);}
      pill(labels[selected.collection]||'知识条目',article);el('h2',word(selected,'title'),article);el('p',word(selected,'abstract'),article,'abstract');
      const reader=el('div',undefined,article,'evidence-reader');reader.hidden=true;
      function citations(item,parent){const refs=el('div',undefined,parent,'knowledge-evidence');for(const [i,id] of (item.evidence_ids||[]).entries()){const node=loaded.evidence_catalog?.[id];button('来源 '+(i+1),refs,async()=>{reader.hidden=false;try{await evidence(reader,node||{},scope.select.value);}catch(error){fail(reader,error,()=>evidence(reader,node||{},scope.select.value));}reader.scrollIntoView({block:'nearest'});},{style:'btn small',disabled:!node});}}
      for(const claim of selected.claims||[]){const box=el('div',undefined,article,'knowledge-claim');pill(labels[claim.status]||claim.status||'待验证',box);el('p',word(claim,'text'),box);citations(claim,box);}
      for(const section of selected.sections||[]){const box=el('section',undefined,article);el('h3',word(section,'heading'),box);el('p',word(section,'body'),box);citations(section,box);}
      article.append(reader);pager(content,offset,list.length,filtered.length,n=>{offset=n;render();},40);
    }
    input.oninput=collection.onchange=()=>{offset=0;if(loaded)render();};await load();
  }
  function pager(root,offset,count,total,change,size){const row=el('div',undefined,root,'projection-pagination');el('span',`展示 ${offset+1}–${offset+count} / ${total}`,row);const controls=el('div',undefined,row);button('上一页',controls,()=>change(Math.max(0,offset-size)),{style:'btn small',disabled:!offset});button('下一页',controls,()=>change(offset+size),{style:'btn small',disabled:offset+count>=total});}
  async function graph(root){
    root.replaceChildren();heading(root,'SEE HOW YOUR KNOWLEDGE CONNECTS','知识图谱','沿着真实关系探索记忆，点开节点查看内容与证据。');let loaded,offset=0,selected;
    const scope=scopePicker(root,()=>load());const search=el('input',undefined,scope.toolbar);search.type='search';search.placeholder='搜索记忆主题与内容';search.setAttribute('aria-label','搜索图谱');
    button('刷新图谱',scope.toolbar,()=>load(),{symbol:'refresh'});const content=el('div',undefined,root);let generation=0;
    async function load(){const seq=++generation;content.replaceChildren();el('p','正在读取知识图谱…',content,'workspace-status');try{const value=await api('graph',{scope:scope.select.value});if(seq!==generation)return;if(!Array.isArray(value.nodes)||!Array.isArray(value.edges))throw Error('服务尚未返回有效的知识图谱。');loaded=value;offset=0;selected=null;render();}catch(error){if(seq===generation)fail(content,error,load);}}
    function render(){content.replaceChildren();projectionStatus(content,loaded);const q=search.value.toLowerCase().trim();const all=loaded.nodes.filter(n=>n.level==='evidence'&&n.evidence_kind==='memory'&&(!q||JSON.stringify(n).toLowerCase().includes(q)));const nodes=all.slice(offset,offset+40);
      if(!nodes.length){empty(content,'暂时没有匹配的记忆节点',all.length?'调整搜索词查看其他记忆。':'记忆整理后会形成可浏览的节点；关系仅展示服务端已提供的内容。',{symbol:'branch'});return;}
      const ids=new Set(nodes.map(n=>n.id));const edges=loaded.edges.filter(e=>ids.has(e.source)&&ids.has(e.target));const allMap=new Map(loaded.nodes.map(n=>[n.id,n]));if(!ids.has(selected?.id))selected=nodes[0];
      const layout=el('div',undefined,content,'graph-layout');const canvas=el('div',undefined,layout,'graph-canvas');const controls=el('div',undefined,canvas,'graph-controls');el('span',`${nodes.length} 个节点 · ${edges.length} 条可见关系`,controls);const zoom=el('div',undefined,controls);
      const ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.classList.add('map');svg.setAttribute('aria-label','记忆关系图，可缩放、拖动并选择节点');canvas.append(svg);
      const make=(tag,attrs,parent=svg)=>{const n=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,String(v));parent.append(n);return n;};
      const defs=make('defs',{});const marker=make('marker',{id:'memory-arrow',viewBox:'0 0 10 10',refX:19,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'},defs);make('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#a9ada2'},marker);
      let view={x:0,y:0,w:1000,h:670};const apply=()=>svg.setAttribute('viewBox',`${view.x} ${view.y} ${view.w} ${view.h}`);const scale=f=>{const w=view.w*f,h=view.h*f;if(w<220||w>4000)return;view={x:view.x+(view.w-w)/2,y:view.y+(view.h-h)/2,w,h};apply();};apply();button('−',zoom,()=>scale(1.25),{style:'btn small'}).setAttribute('aria-label','缩小图谱');button('＋',zoom,()=>scale(.8),{style:'btn small'}).setAttribute('aria-label','放大图谱');button('复位',zoom,()=>{view={x:0,y:0,w:1000,h:670};apply();},{style:'btn small'});
      const cols=Math.max(2,Math.ceil(Math.sqrt(nodes.length*1.5))),rows=Math.ceil(nodes.length/cols);const pos=new Map(nodes.map((n,i)=>[n.id,{x:100+(i%cols)*800/Math.max(1,cols-1),y:90+Math.floor(i/cols)*490/Math.max(1,rows-1)}]));
      // Deterministic layout only. Edges and their direction always come from the API.
      for(let k=0;k<85;k++){const force=new Map(nodes.map(n=>[n.id,{x:0,y:0}]));for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){const a=pos.get(nodes[i].id),b=pos.get(nodes[j].id);let dx=a.x-b.x,dy=a.y-b.y;const d=Math.max(30,Math.hypot(dx,dy));const f=1800/d/d;force.get(nodes[i].id).x+=dx*f;force.get(nodes[i].id).y+=dy*f;force.get(nodes[j].id).x-=dx*f;force.get(nodes[j].id).y-=dy*f;}for(const e of edges){const a=pos.get(e.source),b=pos.get(e.target),dx=b.x-a.x,dy=b.y-a.y;const f=(Math.hypot(dx,dy)-160)*.0009;force.get(e.source).x+=dx*f;force.get(e.source).y+=dy*f;force.get(e.target).x-=dx*f;force.get(e.target).y-=dy*f;}for(const n of nodes){const p=pos.get(n.id),f=force.get(n.id);p.x=Math.min(870,Math.max(110,p.x+Math.max(-5,Math.min(5,f.x))));p.y=Math.min(590,Math.max(70,p.y+Math.max(-5,Math.min(5,f.y))));}}
      for(const edge of edges){const a=pos.get(edge.source),b=pos.get(edge.target);const line=make('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:'graph-edge','marker-end':'url(#memory-arrow)'});make('title',{},line).textContent=labels[edge.type]||edge.type;}
      const groups=new Map();const detail=el('aside',undefined,layout,'panel graph-detail');
      function select(n){selected=n;for(const [id,g]of groups)g.setAttribute('aria-pressed',String(id===n.id));detail.replaceChildren();pill(labels[n.memory_type]||'记忆',detail);el('h2',word(n,'label'),detail);el('p',word(n,'summary'),detail);el('p',`来源角色：${labels[n.actor_role]||n.actor_role||'未标注'}${n.state?' · '+n.state:''}`,detail);const reader=el('div',undefined,detail,'evidence-reader');reader.hidden=true;button('查看原始证据',detail,async()=>{reader.hidden=false;try{await evidence(reader,n,scope.select.value);}catch(error){fail(reader,error,()=>evidence(reader,n,scope.select.value));}},{style:'btn small',symbol:'document'});
        const relations=loaded.edges.filter(e=>(e.source===n.id||e.target===n.id)&&allMap.get(e.source)?.evidence_kind==='memory'&&allMap.get(e.target)?.evidence_kind==='memory');
        for(const edge of relations){const box=el('div',undefined,detail,'graph-relationship');const from=allMap.get(edge.source),to=allMap.get(edge.target);el('strong',`${word(from,'label')} → ${labels[edge.type]||edge.type} → ${word(to,'label')}`,box);if(edge.reason)el('p',edge.reason,box);el('p',edge.origin==='agent'?'模型整理关系，请结合证据核对':'服务端记录关系',box);}
        if(!relations.length)el('p','当前节点尚无已记录的语义关系。',detail,'workspace-status');detail.append(reader);
      }
      for(const n of nodes){const p=pos.get(n.id);const g=make('g',{transform:`translate(${p.x},${p.y})`,class:'graph-node',role:'button',tabindex:0,'aria-label':word(n,'label'),'aria-pressed':String(selected.id===n.id),'data-actor':n.actor_role||'user'});groups.set(n.id,g);make('circle',{r:8},g);const text=make('text',{y:27,'text-anchor':'middle'},g);text.textContent=word(n,'label').slice(0,18);make('title',{},g).textContent=word(n,'label');g.onclick=()=>select(n);g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(n);}};}
      select(selected);let drag;
      svg.onpointerdown=e=>{if(e.target.closest('.graph-node'))return;drag={x:e.clientX,y:e.clientY,base:{...view}};svg.setPointerCapture(e.pointerId);};svg.onpointermove=e=>{if(!drag)return;const r=svg.getBoundingClientRect();view.x=drag.base.x-(e.clientX-drag.x)*view.w/r.width;view.y=drag.base.y-(e.clientY-drag.y)*view.h/r.height;apply();};svg.onpointerup=svg.onpointercancel=()=>drag=null;
      const legend=el('div',undefined,canvas,'graph-legend');for(const t of ['用户来源','助手来源','箭头表示服务端关系方向']){const row=el('span',undefined,legend);if(t!=='箭头表示服务端关系方向')el('i',undefined,row);el('span',t,row);}
      const list=el('div',undefined,canvas,'graph-node-list');list.setAttribute('aria-label','可选择的记忆节点');for(const n of nodes)button(word(n,'label'),list,()=>select(n),{style:'btn small'});
      pager(content,offset,nodes.length,all.length,n=>{offset=n;render();},40);
    }
    search.oninput=()=>{offset=0;if(loaded)render();};await load();
  }
  window.mountWorkspacePage = page => {if(!['providers','knowledge','graph'].includes(page)||pages.get(page))return;pages.set(page,true);const root=$('page-'+page);el('p','正在加载…',root,'workspace-status');const start=()=>({providers,knowledge,graph}[page])(root);void start().catch(error=>fail(root,error,start));};
})();
