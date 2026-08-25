"""POST, GET and DELETE on /v1/webhooks — a tenant's callback registrations.

Every handler passes its tenant into the store rather than filtering what comes back, so
another tenant's registration is absent from a result instead of fetched and then
dropped, and a delete under a stolen id writes nothing rather than writing and then
being reverted.

A read returns the registration exactly as stored, which includes `secret_ref`. That is
a name in the credential vault and not signing material: whoever registered it already
knows it, and it buys nothing without the vault read the dispatcher does at delivery
time.

There is no update route. A registration is four short fields and changing one is a
delete and a register, which is also the honest semantics — an edited destination is a
different destination, and the deliveries already claimed under the old one should not
follow it there.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from managed_agent.control.api.refusals import refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.webhooks.registry import (
    RegisterWebhook,
    WebhookInvalid,
    WebhookView,
    parse_callback_url,
    parse_secret_ref,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import TenantId

router = APIRouter(tags=["webhooks"])


class WebhookList(BaseModel):
    model_config = ConfigDict(frozen=True)

    webhooks: tuple[WebhookView, ...]


@router.post(
    "/webhooks",
    status_code=status.HTTP_201_CREATED,
    response_model=WebhookView,
    responses={STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope}},
)
async def register(
    body: RegisterWebhook,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> WebhookView | JSONResponse:
    """Register a destination and the states it wants. Returns it as stored.

    Both fields are parsed here rather than in the model so their refusals carry a code
    from the closed set in the platform's own envelope, which a field validator's 400
    would not.

    The reference is refused for the same reason the url is: it is a registration only
    the tenant can fix. Left to the delivery path it becomes callbacks that silently
    never arrive, with the reason in a platform log the tenant cannot read -- the
    misattribution running the other way.
    """
    try:
        url = parse_callback_url(body.url)
    except WebhookInvalid as invalid:
        return refuse(ErrorCode.REQUEST_INVALID, invalid.reason, url=body.url)
    try:
        secret_ref = parse_secret_ref(body.secret_ref)
    except WebhookInvalid as invalid:
        # The detail echoes the reference and never the url, so each refusal names the
        # field the tenant has to change. The reference is the tenant's own text and
        # holds no secret -- it is a name in the vault, which is the whole reason a
        # registration carries one instead of a value.
        return refuse(
            ErrorCode.REQUEST_INVALID, invalid.reason, secret_ref=body.secret_ref
        )
    record = await platform_from_request(request).webhooks.register(
        tenant_id, url, body.states, secret_ref
    )
    return WebhookView.of(record)


@router.get("/webhooks", response_model=WebhookList)
async def list_webhooks(
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> WebhookList:
    """Every registration this tenant has, oldest first."""
    records = await platform_from_request(request).webhooks.list_for_tenant(tenant_id)
    return WebhookList(webhooks=tuple(WebhookView.of(r) for r in records))


@router.delete(
    "/webhooks/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    # Explicitly no response model. FastAPI otherwise derives one from the return
    # annotation and then refuses the route outright at import time, because 204 may
    # carry no body -- the refusal on the unhappy path is a JSONResponse of its own and
    # is declared under `responses` instead.
    response_model=None,
    responses={STATUS_FOR[ErrorCode.WEBHOOK_NOT_FOUND]: {"model": PublicErrorEnvelope}},
)
async def unregister(
    webhook_id: UUID,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> Response | JSONResponse:
    """Stop callbacks on one registration, and drop what was recorded against it.

    Refuses rather than answering 204 for an id this tenant does not hold, because a
    tenant that mistyped an id and got 204 would believe it had stopped a callback it
    had not.
    """
    removed = await platform_from_request(request).webhooks.delete(
        webhook_id, tenant_id
    )
    if not removed:
        return refuse(
            ErrorCode.WEBHOOK_NOT_FOUND,
            "no such webhook for this tenant",
            webhook_id=str(webhook_id),
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
